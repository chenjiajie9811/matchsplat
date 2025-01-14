import cv2
from einops import rearrange
from typing import Optional, Tuple
import torch
import numpy as np
from kornia.geometry.epipolar import normalize_points, normalize_transformation, motion_from_essential, \
    motion_from_essential_choose_solution, triangulate_points, symmetrical_epipolar_distance
from kornia.geometry.epipolar.projection import depth_from_point
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.utils import eye_like
from kornia.utils.helpers import _torch_linalg_svdvals
from kornia.core import arange, ones_like, where, zeros

from ..loftr.loftr import LoFTR
import torchvision.transforms as tf
# from pose_optimization.two_view.bundle_adjust_gauss_newton_2_view import BundleAdjustGaussNewton2View
# from pose_optimization.two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle

def compute_rotation_error(T0, T1, reduce=True):
    # use diagonal and sum to compute trace of a batch of matrices
    cos_a = ((T0[..., :3, :3].transpose(-1, -2) @ T1[..., :3, :3]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) \
        - 1.) / 2.
    cos_a = torch.clamp(cos_a, -1., 1.) # avoid nan
    abs_acos_a = torch.abs(torch.arccos(cos_a))
    if reduce:
        return abs_acos_a.mean()
    else:
        return abs_acos_a

def compute_translation_error_as_angle(T0, T1, reduce=True):
    n = torch.linalg.norm(T0[..., :3, 3], dim=-1) * torch.linalg.norm(T1[..., :3, 3], dim=-1)
    valid_n = n > 1e-6
    T0_dot_T1 = (T0[..., :3, 3][valid_n] * T1[..., :3, 3][valid_n]).sum(-1)
    err = torch.abs(torch.arccos((T0_dot_T1 / n[valid_n]).clamp(-1., 1.)))
    if reduce:
        return err.mean()
    else:
        return err

def normalize(kpts, intr):
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = intr[..., 0, 0], intr[..., 1, 1], intr[..., 0, 2], intr[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    n_kpts[..., 1] = (kpts[..., 1] - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
    return n_kpts

def get_kpts(data, result, id0, id1):
    if "keypoints" + str(id0) in data:
        keypoints0, keypoints1 = data["keypoints" + str(id0)], data["keypoints" + str(id1)]
    else:
        keypoints0, keypoints1 = data["keypoints{}_{}_{}".format(id0, id0, id1)], data["keypoints{}_{}_{}".format(id1, id0, id1)]
    matches = result["matches{}_{}_{}".format(id0, id0, id1)]
    intr0, intr1 = data["intr" + str(id0)], data["intr" + str(id1)]
    bs, n_kpts0, _ = keypoints0.shape
    dev = keypoints0.device
    # determine confidence
    batch_idx0 = torch.arange(bs, device=dev).unsqueeze(-1).expand(bs, n_kpts0)
    conf_key = "conf_scores_{}_{}".format(id0, id1)
    confidence = result[conf_key]
    confidence = (matches >= 0).float().unsqueeze(-1) * confidence
    keypoints1 = keypoints1[batch_idx0, matches]
    return keypoints0, keypoints1, intr0, intr1, confidence

# adapted from https://github.com/kornia/kornia
def find_fundamental(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape, points2.shape)
    if not (len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]):
        raise AssertionError(weights.shape)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    X = w_diag @ X

    # compute eigevectors and retrieve the one with the smallest eigenvalue
    _, _, V = torch.svd(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = torch.svd(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

    return normalize_transformation(F_est)

def estimate_relative_pose_w8pt(kpts0, kpts1, intr0, intr1, confidence, choose_closest=False, T_021=None, determine_inliers=False):
    if kpts0.shape[1] < 8:
        return None, None
    sum_conf = confidence.sum(dim=1, keepdim=True) + 1e-6
    confidence = confidence / sum_conf
    kpts0_norm = normalize(kpts0, intr0)
    kpts1_norm = normalize(kpts1, intr1)
    dev = intr0.device
    bs = intr0.shape[0]
    intr = torch.eye(3, device=dev).unsqueeze(0)
    Fs = find_fundamental(kpts0_norm, kpts1_norm, confidence.squeeze(-1))
    if choose_closest:
        Rs, ts = motion_from_essential(Fs)
        min_err = torch.full((bs,), 1e6, device=dev)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        for R, t in zip(Rs.permute(1, 0, 2, 3), ts.permute(1, 0, 2, 3)):
            pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
            pred_T021[:, :3, :3] = R
            pred_T021[:, :3, 3] = t.squeeze(-1)

            curr_err = compute_rotation_error(pred_T021, T_021, reduce=False) + compute_translation_error_as_angle(pred_T021, T_021, reduce=False)
            update_mask = curr_err < min_err

            min_err[update_mask] = curr_err[update_mask]
            min_err_pred_T021[update_mask] = pred_T021[update_mask]
    else:
        R, t, pts = motion_from_essential_choose_solution(Fs, intr, intr, kpts0_norm, kpts1_norm, mask=None)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        min_err_pred_T021[:, :3, :3] = R
        min_err_pred_T021[:, :3, 3] = t.squeeze(-1)
    # check for positive depth
    P0 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)[:, :3, :]
    pts_3d = triangulate_points(P0, min_err_pred_T021[:, :3, :], kpts0_norm, kpts1_norm)
    depth0 = pts_3d[..., -1]
    depth1 = depth_from_point(min_err_pred_T021[:, :3, :3], min_err_pred_T021[:, :3, 3:], pts_3d)
    pos_depth_mask = torch.logical_and(depth0 > 0., depth1 > 0.)
    epi_err = None
    inliers = None
    if determine_inliers:
        epi_err = symmetrical_epipolar_distance(kpts0_norm, kpts1_norm, Fs)
        epi_err_sqrt = epi_err.sqrt()
        thresh = 3. / ((intr0[:, 0, 0] + intr0[:, 1, 1] + intr1[:, 0, 0] + intr1[:, 1, 1]) / 4.)
        inliers = torch.logical_and(pos_depth_mask, epi_err_sqrt <= thresh.unsqueeze(-1))
    info = {"kpts0_norm": kpts0_norm, 
            "kpts1_norm": kpts1_norm, 
            "confidence": confidence, 
            "inliers": inliers, 
            "pos_depth_mask": pos_depth_mask, 
            "pts_3d" : pts_3d}
    return min_err_pred_T021, info

# def run_weighted_8_point(data, result, id0, id1, choose_closest=False, target_T_021=None):
#     match_key = "matches{}_{}_{}".format(id0, id0, id1)
#     if match_key in result and result[match_key].shape[1] != 0:
#         kpts0, kpts1, intr0, intr1, confidence = get_kpts(data, result, id0, id1)
#         return estimate_relative_pose_w8pt(kpts0, kpts1, intr0, intr1, confidence, choose_closest=choose_closest, T_021=target_T_021)
#     else:
#         return None, None

def run_weighted_8_point(batch):
    b, v, c, h, w = batch['context']['image'].shape
    # Only support 2 views and single surface single gaussians for now
    assert v == 2
    mbids = batch["mbids"]
    
    # Estimate the relative pose from 0 to 1
    # Only update the extrinsics of the second camera, the extr of the first camera remain
    # So that we can keep the target views also unchanged
    extrinsics_est = batch["context"]["extrinsics"].clone().detach()
    
    trans_err_abs_list = []
    rot_err_abs_list = []
    trans_err_rel_list = []
    rot_err_rel_list = []
    
    # Perform pose estimation in each batch
    for i in range(b):
        b_mask = mbids == i
        mkpts0 = batch["mkpts0"][b_mask]
        mkpts1 = batch["mkpts1"][b_mask]
        mconf = batch["mconf"][b_mask]

        extr0 = extrinsics_est[i, 0].clone()
        extr1 = extrinsics_est[i, 1].clone()

        intr0 = batch["context"]["intrinsics"][i, 0].clone().detach()
        intr1 = batch["context"]["intrinsics"][i, 1].clone().detach()

        intr0[0, :] *= w
        intr0[1, :] *= h
        intr1[0, :] *= w
        intr1[1, :] *= h

        T_021_gt = torch.inverse(extr1) @ extr0
        scale = torch.norm(T_021_gt[:3, 3])

        T_021_est, info = estimate_relative_pose_w8pt(
            mkpts0[None], mkpts1[None], 
            intr0[None], intr1[None], 
            mconf[None], 
            choose_closest=False, T_021=T_021_gt[None])
        T_021_est = T_021_est.squeeze()
        T_021_est[:3, 3] *= scale

        # update the estimated cam-to-world of the second cam
        # extr1_est = T_021_est @ extr0
        extr1_est =  extr0 @ torch.inverse(T_021_est)
        extrinsics_est[i, 1] = extr1_est.squeeze()

        trans_err_abs_list.append(compute_translation_error_as_angle(extr1_est[None], extr1[None], reduce=False))
        rot_err_abs_list.append(compute_rotation_error(extr1_est[None], extr1[None], reduce=False))
        trans_err_rel_list.append(compute_translation_error_as_angle(T_021_est[None], T_021_gt[None], reduce=False))
        rot_err_rel_list.append(compute_rotation_error(T_021_est[None], T_021_gt[None], reduce=False))
        

        # eval
        # print (f"gt extr1:\n {extr1}")
        # print (f"est extr1:\n {extr1_est}")
        # print (f"rotation err:\n {compute_rotation_error(extr1_est[None], extr1[None], reduce=False)}") 
        # print (f"translation error:\n {compute_translation_error_as_angle(extr1_est[None], extr1[None], reduce=False)}")
        # print ()
        # print (f"gt rel:\n {T_021_gt}")
        # print (f"est rel:\n {T_021_est}")
        # print (f"rotation err:\n {compute_rotation_error(T_021_est[None], T_021_gt[None], reduce=False)}") 
        # print (f"translation error:\n {compute_translation_error_as_angle(T_021_est[None], T_021_gt[None], reduce=False)}")

    pose_eval_dict = {
        "rot_err_abs" : torch.stack(rot_err_abs_list).mean(),
        "trans_err_abs" : torch.stack(trans_err_abs_list).mean(),
        "rot_err_rel" : torch.stack(rot_err_rel_list).mean(),
        "trans_err_rel" : torch.stack(trans_err_rel_list).mean(),
    }

    return extrinsics_est, pose_eval_dict

        

def estimate_relative_pose_cv2(kpts0, kpts1, intr0, intr1):
    E, mask = cv2.findEssentialMat(kpts0, kpts1, intr0, cv2.LMEDS, threshold=0.5)

    _, R, t, mask = cv2.recoverPose(E, kpts0, kpts1, cameraMatrix=intr0)

    kpts0 = kpts0[mask.ravel()==255]
    kpts1 = kpts1[mask.ravel()==255]

    proj0 = np.zeros((3, 4))
    proj0[:3, :3] = np.eye(3)
    proj0  = intr0 @ proj0

    proj1 = np.concatenate([R, t.reshape(3, 1)], axis=1)
    proj1 = intr1 @ proj1

    pts = cv2.triangulatePoints(proj0, proj1, kpts0.T, kpts1.T)
    pts_3d = cv2.convertPointsFromHomogeneous(pts.T).reshape(-1,3)
    return R, t, pts_3d
   

def pose_estimation_multi_view(batch, matcher: LoFTR):
    b, v, c, h, w = batch["target"]["image"].shape
    context_images = batch["context"]["image"][:, :1] # b, 1, c, h, w
    context_images = context_images.repeat(1, v, 1, 1, 1) # b, v, c, h, w

    target_images = batch["target"]["image"] 

    extrinsics_est = batch["target"]["extrinsics"].clone().detach()

    to_gray = tf.Grayscale()

    for i in range(b):
        # Run lofter matching
        with torch.no_grad():
            data = {"image0": to_gray(context_images[i]), "image1": to_gray(target_images[i])}
            matcher(data)

        for j in range(v):
            v_mask = data["m_bids"] == j
            mkpts0 = data["mkpts0_f"][v_mask]
            mkpts1 = data["mkpts1_f"][v_mask]
            mconf = data["mconf"][v_mask]

            conf_mask = mconf >= 0.5
            mkpts0 = mkpts0[conf_mask]
            mkpts1 = mkpts1[conf_mask]
            mconf = mconf[conf_mask]

            # Camera 0 is always the first camera in context of each batch
            extr0 = batch["context"]["extrinsics"][i, 0].clone().detach()
            extr1 = batch["target"]["extrinsics"][i, j].clone().detach()

            intr0 = batch["context"]["intrinsics"][i, 0].clone().detach()
            intr1 = batch["target"]["intrinsics"][i, j].clone().detach()

            intr0[0, :] *= w
            intr0[1, :] *= h
            intr1[0, :] *= w
            intr1[1, :] *= h

            T_021_gt = torch.inverse(extr1) @ extr0
            scale = torch.norm(T_021_gt[:3, 3])

            T_021_est, info = estimate_relative_pose_w8pt(
                mkpts0[None], mkpts1[None], 
                intr0[None], intr1[None], 
                mconf[None], 
                choose_closest=False, T_021=T_021_gt[None])
            T_021_est = T_021_est.squeeze()
            T_021_est[:3, 3] *= scale

            # update the estimated cam-to-world of the second cam
            # extr1_est = T_021_est @ extr0
            extr1_est =  extr0 @ torch.inverse(T_021_est)
            extrinsics_est[i, j] = extr1_est.squeeze()

            # eval
            print (f"gt extr1:\n {extr1}")
            print (f"est extr1:\n {extr1_est}")
            print (f"rotation err:\n {compute_rotation_error(extr1_est[None], extr1[None], reduce=False)}") 
            print (f"translation error:\n {compute_translation_error_as_angle(extr1_est[None], extr1[None], reduce=False)}")
            print ()
            print (f"gt rel:\n {T_021_gt}")
            print (f"est rel:\n {T_021_est}")
            print (f"rotation err:\n {compute_rotation_error(T_021_est[None], T_021_gt[None], reduce=False)}") 
            print (f"translation error:\n {compute_translation_error_as_angle(T_021_est[None], T_021_gt[None], reduce=False)}")


    return extrinsics_est


# def run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence, init_T021, n_iterations, check_lu_info_strict=False, \
#         check_precond_strict=False):
#     bs = kpts0_norm.shape[0]
#     ba = BundleAdjustGaussNewton2View(batch_size=bs, n_iterations=n_iterations, check_lu_info_strict=check_lu_info_strict, \
#                                       check_precond_strict=check_precond_strict)
#     extrinsics, valid_batch = ba.run(kpts0_norm, kpts1_norm, confidence.squeeze(-1), init_T021)
#     return extrinsics[:, 1], valid_batch

################################################################
# Weighted Perspective-N-Point
# Adapted from kornia
################################################################

def _mean_isotropic_scale_normalize(points: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalizes points.

    Args:
       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.
       eps : Small value to avoid division by zero error.

    Returns:
       Tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix
       in the shape :math:`(B, D+1, D+1)`.
    """
    if not isinstance(points, torch.Tensor):
        raise AssertionError(f"points is not an instance of torch.Tensor. Type of points is {type(points)}")

    if len(points.shape) != 3:
        raise AssertionError(f"points must be of shape (B, N, D). Got shape {points.shape}.")

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1xD
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B

    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)  # B
    transform = eye_like(D_int + 1, points)  # (B, D+1, D+1)

    idxs = arange(D_int, dtype=torch.int64, device=points.device)
    transform[:, idxs, idxs] = transform[:, idxs, idxs] * scale[:, None]
    transform[:, idxs, D_int] = transform[:, idxs, D_int] + (-scale[:, None] * x_mean[:, 0, idxs])

    points_norm = transform_points(transform, points)  # BxNxD

    return (points_norm, transform)


def solve_pnp_dlt_weighted(
    world_points: torch.Tensor,
    img_points: torch.Tensor,
    intrinsics: torch.Tensor,
    weights: torch.Tensor,
    svd_eps: float = 1e-4,
)-> torch.Tensor:
    """
        Args:
            world_points : A tensor with shape :math:`(B, N, 3)` representing
            the points in the world space.
            img_points : A tensor with shape :math:`(B, N, 2)` representing
            the points in the image space.
            intrinsics : A tensor with shape :math:`(B, 3, 3)` representing
            the intrinsic matrices.
            weights : A tensor with shape : math:'(B, N)' representing 
            the 3D 2D matching confidence
            svd_eps : A small float value to avoid numerical precision issues.

        Returns:
            A tensor with shape :math:`(B, 3, 4)` representing the estimated world to
            camera transformation matrices (also known as the extrinsic matrices).
    """
    B, N = world_points.shape[:2]

    # Getting normalized world points.
    world_points_norm, world_transform_norm = _mean_isotropic_scale_normalize(world_points)

    # Checking if world_points_norm (of any element of the batch) has rank = 3. This
    # function cannot be used if all world points (of any element of the batch) lie
    # on a line or if all world points (of any element of the batch) lie on a plane.
    s = _torch_linalg_svdvals(world_points_norm)
    if torch.any(s[:, -1] < svd_eps):
        raise AssertionError(
            "The last singular value of one/more of the elements of the batch is smaller "
            f"than {svd_eps}. This function cannot be used if all world_points (of any "
            "element of the batch) lie on a line or if all world_points (of any "
            "element of the batch) lie on a plane."
        )

    intrinsics_inv = torch.inverse(intrinsics)
    world_points_norm_h = convert_points_to_homogeneous(world_points_norm)

    # Transforming img_points with intrinsics_inv to get img_points_inv
    img_points_inv = transform_points(intrinsics_inv, img_points)

    # Normalizing img_points_inv
    img_points_norm, img_transform_norm = _mean_isotropic_scale_normalize(img_points_inv)
    inv_img_transform_norm = torch.inverse(img_transform_norm)

    # Setting up the system (the matrix A in Ax=0)
    system = zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = world_points_norm_h
    system[:, 1::2, 4:8] = world_points_norm_h
    system[:, 0::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 0:1]
    system[:, 1::2, 8:12] = world_points_norm_h * (-1) * img_points_norm[..., 1:2]

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    system = w_diag @ system

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)

    # Creating solution_4x4
    solution_4x4 = eye_like(4, solution)
    solution_4x4[:, :3, :] = solution

    # De-normalizing the solution
    intermediate = torch.bmm(solution_4x4, world_transform_norm)
    solution = torch.bmm(inv_img_transform_norm, intermediate[:, :3, :])

    # We obtained one solution for each element of the batch. We may
    # need to multiply each solution with a scalar. This is because
    # if x is a solution to Ax=0, then cx is also a solution. We can
    # find the required scalars by using the properties of
    # rotation matrices. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the all the rotation matrices are non-negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(solution[:, :3, :3])
    ones = ones_like(det)
    sign_fix = where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply solution with mul_factor.
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    temp = solution * mul_factor

    # To make sure that the rotation matrix would be orthogonal, we apply
    # QR decomposition.
    ortho, right = torch.linalg.qr(temp[:, :3, :3])

    # We may need to fix the signs of the columns of the ortho matrix.
    # If right[i, j, j] is negative, then we need to flip the signs of
    # the column ortho[i, :, j]. The below code performs the necessary
    # operations in a better way.
    mask = eye_like(3, ortho)
    col_sign_fix = torch.sign(mask * right)
    rot_mat = torch.bmm(ortho, col_sign_fix)

    # Preparing the final output.
    pred_world_to_cam = torch.cat([rot_mat, temp[:, :3, 3:4]], dim=-1)

    # TODO: Implement algorithm to refine the solution.

    return pred_world_to_cam











    





