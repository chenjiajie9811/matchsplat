from dataclasses import dataclass

import torch
import numpy as np
from einops import reduce
from jaxtyping import Float, Union
from torch import Tensor
import torch.utils

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement
from torchvision.utils import save_image
from PIL import Image
from einops.einops import rearrange

from ..geometry.projection import get_world_rays, sample_image_grid

def save_points_ply(points, path):
    vertex = np.array([(points[i][0], points[i][1], points[i][2]) 
                        for i in range(points.shape[0])], 
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    vertex_element = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(path)

def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz

def iproj_full_img(depth, intr):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    ht, wd = depth.shape[-2:]

    y, x = torch.meshgrid(
        torch.arange(ht).to(depth.device).float(),
        torch.arange(wd).to(depth.device).float())

    i = torch.ones_like(depth)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i], dim=-1) * depth.unsqueeze(-1)
    # pts = torch.stack([pts, i], dim=-1)
    return pts.reshape(-1, 3)

def iproj_full_img_(depth: Tensor, intr: Tensor, extr: Tensor):
    ht, wd = depth.shape[-2:]
    depth = depth.reshape(1, -1) 
    xy_ray, _ = sample_image_grid(((ht, wd)), depth.device)
    xy_ray = xy_ray.reshape(1, -1, 2)
    origins, directions = get_world_rays(xy_ray, extr, intr)
    return origins + directions * depth[..., None]

def iproj_full_img_identity_depth(depth: Tensor, intr: Tensor, extr: Tensor):
    ht, wd = depth.shape[-2:]
    depth = depth.reshape(1, -1) 
    xy_ray, _ = sample_image_grid(((ht, wd)), depth.device)
    xy_ray = xy_ray.reshape(1, -1, 2)
    origins, directions = get_world_rays(xy_ray, extr, intr)
    return origins + directions #* depth[..., None]

# @TODO: Move these geometry related functions to /geometry/projection or check whether we can reuse
def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)


def iproj_kpts(
        depth: Float[Tensor, "h w"], 
        intr: Float[Tensor, "3 3"], 
        mkpts: Float[Tensor, "N 2"]
    )-> Float[Tensor, "N 3"]:

    z = depth[mkpts[:, 1].long().squeeze(), mkpts[:, 0].long().squeeze()]
    kpts_3d_0 = to_homogeneous(mkpts) @ torch.inverse(intr).transpose(-1, -2)
    kpts_3d_0 = kpts_3d_0 * z.view(-1, 1)

    return kpts_3d_0

def pc_transform(pts3d_i: Float[Tensor, "N 3"], Tij: Float[Tensor, "4 4"])-> Float[Tensor, "N 3"]:
    # Tij = Tij.transpose(-1, -2)
    # pts3d_i = pts3d_i @ Tij[:3, :3]
    # pts3d_i = pts3d_i  + Tij[:3, -1:].transpose(0,1).repeat(pts3d_i.size()[0],1)
    # return pts3d_i
    return from_homogeneous(to_homogeneous(pts3d_i) @ Tij.transpose(-1, -2))

def proj(pts3d: Float[Tensor, "N 3"], intr: Float[Tensor, "3 3"])->Float[Tensor, "N 2"]:
    return from_homogeneous(pts3d @ intr.transpose(-1, -2))

def projective_transform(
        T01: Float[Tensor, "4 4"], 
        depth0: Float[Tensor, "h w"], 
        intr0: Float[Tensor, "3 3"], 
        intr1: Float[Tensor, "3 3"], 
        mkpts0: Float[Tensor, "N 2"],
        return_kpts_pc=False):
    X0 = iproj_kpts(depth0, intr0, mkpts0)
    X1 = pc_transform(X0, T01)
    x1 = proj(X1, intr1)
    if return_kpts_pc:
        return x1, X0
    return x1

def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = from_homogeneous(point_3d_homo.unsqueeze(0))[0]

    return point_3d

def huber_loss(pred: torch.Tensor, label: torch.Tensor, reduction: str='none'):
        return torch.nn.functional.huber_loss(pred, label, reduction=reduction) 


@dataclass
class LossCorresCfg:
    weight_repro: float
    weight_depth: float
    use_corres_depth: bool
    use_corres_repro: bool


@dataclass
class LossCorresCfgWrapper:
    corres: LossCorresCfg

class LossCorres(Loss[LossCorresCfg, LossCorresCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        depths = batch["context"]["rendered_depth"]
        depths_gaussians = batch["context"]["est_depth"].clone().squeeze()
        mbids = batch["mbids"]

        b, v, h, w = depths.shape
        # Only supported for 2 views and single surface single gaussians for now
        assert v == 2
        # assert srf == 1 and s == 1

        loss = 0.
    
        loss_repro = 0.
        loss_depth = 0.

        # For now we iter over the batch to extract the matched keypoints with different size
        for i in range(b):
            b_mask = mbids == i
            mkpts0 = batch["mkpts0"][b_mask]
            mkpts1 = batch["mkpts1"][b_mask]
            mconf = batch["mconf"][b_mask]

            # print ("extrinsics shape", batch["context"]["extrinsics"].shape)
            extr0 = batch["context"]["extrinsics"][i, 0].clone().detach()
            extr1 = batch["context"]["extrinsics"][i, 1].clone().detach()

            intr0 = batch["context"]["intrinsics"][i, 0].clone().detach()
            intr1 = batch["context"]["intrinsics"][i, 1].clone().detach()

            intr0[0, :] *= w
            intr0[1, :] *= h
            intr1[0, :] *= w
            intr1[1, :] *= h

            img0 = batch['context']['image'][i, 0]
            img1 = batch['context']['image'][i, 1]
            img0_numpy = img0.clone().detach().cpu().numpy().transpose(1, 2, 0)
            img1_numpy = img1.clone().detach().cpu().numpy().transpose(1, 2, 0)

            depth0 = depths[i, 0].clone().view(h, w)
            depth1 = depths[i, 1].clone().view(h, w)

            # depth0 = 1. / depth0
            # depth1 = 1. / depth1

            if self.cfg.use_corres_repro:
                T01 = torch.inverse(extr1) @ extr0
                T10 = torch.inverse(extr0) @ extr1

                # Reprojection error with the estimated depth from 0->1 and 1->0
                x0_1, X0 = projective_transform(T01, depth0, intr0, intr1, mkpts0, True)
                x1_0, X1 = projective_transform(T10, depth1, intr1, intr0, mkpts1, True)

                
                mask_0 = (x0_1[:, 0] >= 0) & (x0_1[:, 0] < w) & (x0_1[:, 1] >= 0) & (x0_1[:, 1] < h)
                mask_1 = (x1_0[:, 0] >= 0) & (x1_0[:, 0] < w) & (x1_0[:, 1] >= 0) & (x1_0[:, 1] < h)
                
                # @TODO Add the mask for filtering out those with large reprojection error 
                loss0_1 = torch.sum(mconf[mask_0] * torch.norm(x0_1[mask_0] - mkpts1[mask_0], dim=-1))
                loss1_0 = torch.sum(mconf[mask_1] * torch.norm(x1_0[mask_1] - mkpts0[mask_1], dim=-1))
                loss_repro += (loss0_1 + loss1_0) / 2.

                if global_step % 50 == 0:
                    x0_1, mkpts0 = x0_1[mask_0], mkpts0[mask_0]
                    x1_0, mkpts1 = x1_0[mask_1], mkpts1[mask_1]
                    pts3d_0 = iproj_full_img(depth0, intr0)
                    perm = torch.randperm(pts3d_0.size(0))
                    idx = perm[:1000]
                    pts3d_0 = pts3d_0[idx]
                   
                    plt.figure()
                    fig, ax = plt.subplots(nrows=1, ncols=2)
                    ax[0].imshow(img0_numpy)
                    ax[0].scatter(x1_0[:, 0].clone().detach().cpu().numpy(), x1_0[:, 1].clone().detach().cpu().numpy(), s=1, c='red')
                    ax[0].scatter(mkpts0[:, 0].clone().detach().cpu().numpy(), mkpts0[:, 1].clone().detach().cpu().numpy(), s=1, c='blue')
                    
                    ax[1].imshow(img1_numpy)
                    ax[1].scatter(x0_1[:, 0].clone().detach().cpu().numpy(), x0_1[:, 1].clone().detach().cpu().numpy(), s=1, c='red')
                    ax[1].scatter(mkpts1[:, 0].clone().detach().cpu().numpy(), mkpts1[:, 1].clone().detach().cpu().numpy(), s=1, c='blue')
                    plt.savefig(f"outputs/tmp/reproj_120_021_batch{i}_step{global_step}.png")
                    # input()

                # if global_step % 10 == 0:

                #     pts3d_0 = iproj_full_img_(depth0, intr0.unsqueeze(0), extr0.inverse().unsqueeze(0)).squeeze(0)
                #     perm = torch.randperm(pts3d_0.size(0))
                #     idx = perm[:1000]
                #     pts3d_0 = pts3d_0[idx]

                #     pts3d_1 = iproj_full_img_(depth1, intr1.unsqueeze(0), extr1.inverse().unsqueeze(0)).squeeze(0)
                #     perm = torch.randperm(pts3d_1.size(0))
                #     idx = perm[:1000]
                #     pts3d_1 = pts3d_1[idx]

                #     pts3d_0_identity = iproj_full_img_identity_depth(depth0, intr0.unsqueeze(0), extr0.unsqueeze(0)).squeeze(0)
                #     perm = torch.randperm(pts3d_0_identity.size(0))
                #     idx = perm[:1000]
                #     pts3d_0_identity = pts3d_0_identity[idx]

                #     pts3d_1_identity = iproj_full_img_identity_depth(depth1, intr1.unsqueeze(0), extr1.unsqueeze(0)).squeeze(0)
                #     perm = torch.randperm(pts3d_1_identity.size(0))
                #     idx = perm[:1000]
                #     pts3d_1_identity = pts3d_1_identity[idx]

                #     # transform both identity to coord 1
                #     pts3d_0_identity_0 = pts3d_0_identity
                #     save_points_ply(pts3d_0_identity_0, f"outputs/tmp/iproj_identity_0.ply")
                #     pts3d_1_identity_0 = pc_transform(pts3d_1_identity, extr0 @ extr1.inverse())
                #     save_points_ply(pts3d_1_identity_0, f"outputs/tmp/iproj_identity_1.ply")
        
                #     import open3d as o3d
                #     identity0 = o3d.io.read_point_cloud(f"outputs/tmp/iproj_identity_0.ply")
                #     identity1 = o3d.io.read_point_cloud(f"outputs/tmp/iproj_identity_1.ply")

                #     identity0.paint_uniform_color([1, 0, 0]) 
                #     identity1.paint_uniform_color([0, 0, 1])

                #     identity_combined = identity0 + identity1
                #     o3d.io.write_point_cloud(f"outputs/tmp/iproj_identity_combined.ply", identity_combined)


                #     pts3d_0_0 = pts3d_0
                #     save_points_ply(pts3d_0_0, f"outputs/tmp/iproj_0.ply")
                #     pts3d_1_0 = pc_transform(pts3d_1, extr0 @ extr1.inverse())
                #     save_points_ply(pts3d_1_0, f"outputs/tmp/iproj_1.ply")

                #     i0 = o3d.io.read_point_cloud(f"outputs/tmp/iproj_0.ply")
                #     i1 = o3d.io.read_point_cloud(f"outputs/tmp/iproj_1.ply")

                #     i0.paint_uniform_color([1, 0, 0])
                #     i1.paint_uniform_color([0, 0, 1])

                #     i_combined = i0 + i1
                #     o3d.io.write_point_cloud(f"outputs/tmp/iproj_combined.ply", i_combined)


                #     # pts3d_0 = iproj_full_img(depth0, intr0)
                #     # perm = torch.randperm(pts3d_0.size(0))
                #     # idx = perm[:1000]
                #     # pts3d_0 = pts3d_0[idx]

                #     # pts3d_1 = iproj_full_img(depth1, intr1)
                #     # perm = torch.randperm(pts3d_1.size(0))
                #     # idx = perm[:1000]
                #     # pts3d_1 = pts3d_1[idx]

                #     pts3d_0_1 = pc_transform(pts3d_0, T10)
                #     pts3d_both_in_1 = torch.cat([pts3d_0_1, pts3d_1], dim=-0).detach().cpu().numpy()

                #     pts3d_both_in_world = torch.cat([pts3d_0, pts3d_1, pts3d_0_identity, pts3d_1_identity], dim=-0).detach().cpu().numpy()

                #     print (pts3d_0.shape)
                #     print (pts3d_1.shape)
                #     print (pts3d_both_in_world.shape)
                #     save_points_ply(pts3d_both_in_world, f"outputs/tmp/iproj_full_in1_batch{i}_step{global_step}.ply")
                #     save_points_ply(pts3d_0.clone().detach().cpu().numpy(), f"outputs/tmp/iproj_full0_batch{i}_step{global_step}.ply")
                #     save_points_ply(pts3d_1.clone().detach().cpu().numpy(), f"outputs/tmp/iproj_full1_batch{i}_step{global_step}.ply")
                #     plt.figure()
                #     plt.imshow(img0_numpy)
                #     plt.savefig(f"outputs/tmp/img0_batch{i}_step{global_step}.png")
                #     plt.figure()
                #     plt.imshow(img1_numpy)
                #     plt.savefig(f"outputs/tmp/img1_batch{i}_step{global_step}.png")

                #     # save_image(depth0_viz.long(), f"outputs/tmp/depth0_batch{i}_step{global_step}.png")
                #     # save_image(depth1_viz.long(), f"outputs/tmp/depth1_batch{i}_step{global_step}.png")

                #     plt.figure()
                #     fig, ax = plt.subplots(nrows=1, ncols=2)
                #     ax[0].imshow(img0_numpy)
                #     ax[0].scatter(x1_0[:, 0].clone().detach().cpu().numpy(), x1_0[:, 1].clone().detach().cpu().numpy(), s=1, c='red')
                #     ax[0].scatter(mkpts0[:, 0].clone().detach().cpu().numpy(), mkpts0[:, 1].clone().detach().cpu().numpy(), s=1, c='blue')
                #     ax[1].imshow(img1_numpy)
                #     ax[1].scatter(x0_1[:, 0].clone().detach().cpu().numpy(), x0_1[:, 1].clone().detach().cpu().numpy(), s=1, c='red')
                #     ax[1].scatter(mkpts1[:, 0].clone().detach().cpu().numpy(), mkpts1[:, 1].clone().detach().cpu().numpy(), s=1, c='blue')
                #     plt.savefig(f"outputs/tmp/reproj_120_021_batch{i}_step{global_step}.png")

                #     save_points_ply(X0.clone().detach().cpu().numpy(), f"outputs/tmp/iproj_kpts0_batch{i}_step{global_step}.ply")
                #     save_points_ply(X1.clone().detach().cpu().numpy(), f"outputs/tmp/iproj_kpts1_batch{i}_step{global_step}.ply")

                #     input()


            if self.cfg.use_corres_depth:
                # Compute "gt" depth with triangulation, estimated point cloud in world coordinate
                proj_mat0 = torch.matmul(intr0, extr0.inverse()[:3])
                proj_mat1 = torch.matmul(intr1, extr1.inverse()[:3])
                proj_mats = torch.stack([proj_mat0, proj_mat1], dim=0)

                pts_3d = []
                for i in range(len(mkpts0)):
                    pt3d = triangulate_point_from_multiple_views_linear_torch(
                        proj_mats, torch.stack([mkpts0[i], mkpts1[i]]), torch.stack([mconf[i], mconf[i]]))
                    pts_3d.append(pt3d)

                pts_3d_world = torch.stack(pts_3d)

                # Transform from world to the coordinate 0 and 1 and extract the gt depth for the mkpts
                pts_3d_0 = pc_transform(pts_3d_world, extr0.inverse())
                pts_3d_1 = pc_transform(pts_3d_world, extr1.inverse())

                
                kpts_gt_depth_0 = pts_3d_0[:, 2]
                kpts_gt_depth_1 = pts_3d_1[:, 2]

                # Extract the estimated depth for the mkpts
                kpts_est_depth_0 = depth0[mkpts0[:, 1].long().squeeze(), mkpts0[:, 0].long().squeeze()]
                kpts_est_depth_1 = depth0[mkpts1[:, 1].long().squeeze(), mkpts1[:, 0].long().squeeze()]

                # Compute loss by comparing the ratio with 1
                loss_d_0 = torch.sum(torch.abs((kpts_est_depth_0 / (kpts_gt_depth_0 + 1e-6)) - 1.))
                loss_d_1 = torch.sum(torch.abs((kpts_est_depth_1 / (kpts_gt_depth_1 + 1e-6)) - 1.))

                loss_depth += (loss_d_0 + loss_d_1) / 2.

                if global_step % 10 == 0:
                    save_points_ply(pts_3d_0.clone().detach().cpu().numpy(), f"outputs/tmp/triangulated_kpts0_batch{i}_step{global_step}.ply")
                    save_points_ply(pts_3d_1.clone().detach().cpu().numpy(), f"outputs/tmp/triangulated_kpts1_batch{i}_step{global_step}.ply")




        # Devided by the number of all matched keypoints
        loss += self.cfg.weight_repro * (loss_repro / float(len(mkpts0)))
        loss += self.cfg.weight_depth * (loss_depth / float(len(mkpts0)))


        return loss

        
