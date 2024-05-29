import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel

import numpy as np
from scipy.stats import lognorm

from ....geometry.projection import (
    pc_transform,
    save_points_ply, save_color_points_ply,
    triangulate_point_from_multiple_views_linear_torch,
    iproj_full_img_
)


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature

def get_depth_guided_candi(near, far, far_mk, num_samples, sigma=0.8):
    mk_depth_candi = []
    max_depth_mk = rearrange(1.0 / far_mk.cpu().detach(), "b v -> (v b) 1")
    max_depth_mk = rearrange(far_mk.cpu().detach(), "b v -> (v b) 1")
    max_depth_gt = rearrange(far.cpu().detach(), "b v -> (v b) 1")
    min_depth_gt = rearrange(near.cpu().detach(), "b v -> (v b) 1")
    
    for j in range(max_depth_mk.size()[0]):
        mean = max_depth_mk[j]
        scale = mean / np.exp(0.5 * sigma**2)  
        x = np.linspace(int(min_depth_gt[j]), int(max_depth_gt[j]), \
                        int(max_depth_gt[j]) - int(min_depth_gt[j]))
        pdf = lognorm.pdf(x, sigma, scale=scale)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]  
        uniform_samples = np.random.uniform(0, 1, num_samples)
        sample = np.interp(uniform_samples, cdf, x)
        sample = torch.tensor(np.sort(sample, axis=0))
        mk_depth_candi.append(sample)
    mk_depth_candi = torch.stack(mk_depth_candi, 0)
    return mk_depth_candi


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, 
    near, far, num_samples, 
    near_mk, far_mk, depth_guided_sample
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    if depth_guided_sample:
        mk_depth_candi = get_depth_guided_candi(near, far, far_mk, num_samples)
        mk_depth_candi = (1.0 / mk_depth_candi).type_as(features) # depth -> disp
        depth_candi_curr = repeat(mk_depth_candi, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    else:
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
        max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        ).type_as(features)
        depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr

def estimate_depth_from_mkpt(features, intrinsics, extrinsics, batch, 
                             save_ply=False, min_th=0.0, th_max=500.0):
    b, v, c, h, w = features.shape
    mbids = batch["mbids"]
    depth_mkpt_list = []
    near_mks, far_mks =[], []
    
    pc_mkpt_world_list = []
    
    for i in range(b):
        b_mask = mbids == i
        mkpts0 = batch["mkpts0"][b_mask]
        mkpts1 = batch["mkpts1"][b_mask]
        mconf = batch["mconf"][b_mask]

        extr0 = extrinsics[i, 0].clone().detach()
        extr1 = extrinsics[i, 1].clone().detach()
        intr0 = intrinsics[i, 0].clone().detach()
        intr1 = intrinsics[i, 1].clone().detach()

        intr0[0, :] *= w
        intr0[1, :] *= h
        intr1[0, :] *= w
        intr1[1, :] *= h

        proj_mat0 = torch.matmul(intr0, extr0.inverse()[:3])
        proj_mat1 = torch.matmul(intr1, extr1.inverse()[:3])
        proj_mats = torch.stack([proj_mat0, proj_mat1], dim=0)

        pts_3d = []
        for k in range(len(mkpts0)):
            pt3d = triangulate_point_from_multiple_views_linear_torch(
                proj_mats, torch.stack([mkpts0[k], mkpts1[k]]), torch.stack([mconf[k], mconf[k]]))
            pts_3d.append(pt3d)
        pts_3d_world = torch.stack(pts_3d)

        pc_mkpt_world_list.append(pts_3d_world)
        
        # Transform from world to the coordinate 0 and 1
        pts_3d_0 = pc_transform(pts_3d_world, extr0.inverse())
        pts_3d_1 = pc_transform(pts_3d_world, extr1.inverse())

        """ 
        # for test img
        img0 = batch['context']['image'][i, 0]
        img1 = batch['context']['image'][i, 1]
        img0_numpy = img0.clone().detach().cpu().numpy().transpose(1, 2, 0)
        img1_numpy = img1.clone().detach().cpu().numpy().transpose(1, 2, 0)
        plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(img0_numpy)
        ax[1].imshow(img1_numpy)
        plt.savefig(f"outputs/tmp/tt.png")
        """
        
        mask = (pts_3d_0[:, 2] > min_th) & (pts_3d_0[:, 2] < th_max) & \
            (pts_3d_1[:, 2] > min_th) & (pts_3d_1[:, 2] < th_max) 
        pts_3d_0, pts_3d_1 = pts_3d_0[mask], pts_3d_1[mask]
        mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]
        
        if save_ply:
            save_points_ply(pts_3d_0.clone().detach().cpu().numpy(), f"outputs/tmp/tri0_b{i}.ply")
            save_points_ply(pts_3d_1.clone().detach().cpu().numpy(), f"outputs/tmp/tri1_b{i}.ply")
        
        # [K,3] -> [V, K, 3] -> [B V K 3]
        pts_2d_0 = torch.cat([mkpts0, pts_3d_0[:, 2].unsqueeze(1)], dim=1)
        pts_2d_1 = torch.cat([mkpts1, pts_3d_1[:, 2].unsqueeze(1)], dim=1)
        pts_2d = torch.stack([pts_2d_0, pts_2d_1], dim=0) # [v k 3]
        depth_mkpt_list.append(pts_2d)

        # Get a rough range of depth
        near_mk, _ = torch.min(pts_2d[:, :, 2], dim=1) # [v]
        far_mk, _  = torch.max(pts_2d[:, :, 2], dim=1)
        near_mks.append(near_mk.unsqueeze(0))
        far_mks.append(far_mk.unsqueeze(0))

    near_mks = torch.cat(near_mks, dim=0) # [B V]
    far_mks = torch.cat(far_mks, dim=0) # [B V]
    return depth_mkpt_list, near_mks, far_mks, pc_mkpt_world_list

def fusion_mkpt_costvolum(b, depth_mkpt, fullres_disps, alpha=None):
    """
    :param depth_mkpt:    len(list)=b, (v, k, 3)
    :param fullres_disps: (vb, 1, h, w)
    :param alpha: None | float
    :return: (vb, 1, h, w)
    """
    b = len(depth_mkpt)
    v, _, _ = depth_mkpt[0].size() # v k 3                
    fullres_disps = rearrange(fullres_disps, "(v b) ... -> b v ...", b=b, v=v) 
    fullres_disps = fullres_disps.squeeze(2) # b v h w 
    
    if False: # just for testing disp distribution 
        mk_x, mk_y = depth_mkpt[0][:, :, 0].long(),  depth_mkpt[0][:, :, 1].long()
        mk_z = 1.0 / depth_mkpt[0][:, :, 2]
        Xmk = mk_x[0].cpu().detach().numpy()
        Ymk = mk_y[0].cpu().detach().numpy()
        Zmk = mk_z[0].cpu().detach().numpy()
        xx, yy = np.arange(256), np.arange(256)
        xx, yy = np.meshgrid(xx, yy)
        zz = fullres_disps[0,0].cpu().detach().numpy()
                                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xmk, Ymk, Zmk, c='r',label='match')
        ax.scatter(xx, yy, zz, c='b',label='costvolume')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=30, azim=45)
        plt.legend()
        plt.savefig('outputs/tmp/disp_mk_cost_v0.jpg')
        # plt.show()
    
    for i in range(b):
        mk_x, mk_y = depth_mkpt[i][:, :, 0].long(),  depth_mkpt[i][:, :, 1].long()
        mk_z = 1.0 / depth_mkpt[i][:, :, 2]
        for vi in range(v):
            index = (mk_y[vi], mk_x[vi])
            if alpha == None:
                fullres_disps[i, vi].index_put_(index, mk_z[vi])
            else:
                cost_depth = fullres_disps[i, vi, mk_y[vi], mk_x[vi]]
                weight_depth = alpha * mk_z[vi] + (1 - alpha) * cost_depth
                fullres_disps[i, vi].index_put_(index, weight_depth)
    fullres_disps = rearrange(fullres_disps, "b v ... -> (v b) 1 ...", b=b, v=v) 
    return fullres_disps    

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=[256, 128, 64], # input channel # 128
        upscale_factor=[8, 4, 2], # 4
        num_depth_candidates=128,
        costvolume_unet_feat_dim=[256, 128, 64], # 128

        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine

        for i in range(len(feature_channels)):
            # Cost volume refinement: 2D U-Net
            input_channels = feature_channels[i] if wo_cost_volume else (num_depth_candidates + feature_channels[i])
            channels = self.regressor_feat_dim[i]
            
            if wo_cost_volume_refine:
                corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
                setattr(self, "corr_project" + str(i), corr_project)
            else:
                modules = [
                    nn.Conv2d(input_channels, channels, 3, 1, 1),
                    nn.GroupNorm(8, channels),
                    nn.GELU(),
                    UNetModel(
                        image_size=None,
                        in_channels=channels,
                        model_channels=channels,
                        out_channels=channels,
                        num_res_blocks=1,
                        attention_resolutions=costvolume_unet_attn_res,
                        channel_mult=costvolume_unet_channel_mult,
                        num_head_channels=32,
                        dims=2,
                        postnorm=True,
                        num_frames=num_views,
                        use_cross_view_self_attn=True,
                    ),
                    nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
                ]

                corr_refine_net = nn.Sequential(*modules)
                setattr(self, "corr_refine_net" + str(i), corr_refine_net)
                # cost volume u-net skip connection
                regressor_residual = nn.Conv2d(
                    input_channels, num_depth_candidates, 1, 1, 0
                )
                setattr(self, "regressor_residual" + str(i), regressor_residual)

        # Cost Volume Fusion layers
        # 1->1/8 2->1/4 3->1/2 
        self.layer1_outconv = conv1x1(num_depth_candidates, num_depth_candidates)
        self.layer2_outconv = conv1x1(num_depth_candidates, num_depth_candidates)
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(num_depth_candidates, num_depth_candidates),
            nn.LeakyReLU(),
            conv3x3(num_depth_candidates, num_depth_candidates),
        )
        self.layer3_outconv = conv1x1(num_depth_candidates, num_depth_candidates)
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(num_depth_candidates, num_depth_candidates),
            nn.LeakyReLU(),
            conv3x3(num_depth_candidates, num_depth_candidates),
            nn.LeakyReLU(),
        )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        self.fuser1 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, 1, 1),
            nn.GELU(),
        )

        self.upsampler1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(
                scale_factor=2.,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        self.fuser2 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, 1, 1),
            nn.GELU(),
        )

        self.upsampler2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(
                scale_factor=2.,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        self.fuser3 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1),
            nn.GELU(),
        )

        self.upsampler3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Upsample(
                scale_factor=2.,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        self.proj_feature = nn.Conv2d(
            64, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1 # 64+5
        channels = depth_unet_feat_dim

        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels[-1] #64 + 3 + 128
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)

    def trans_cnn_fpn_fusion(self, ts, cs):
        """
            ([bv, 256, 32, 32]), ([bv, 128, 64, 64]), ([bv, 64, 128, 128])
        """
        t1, t2, t3 = ts[0], ts[1], ts[2]
        c1, c2, c3 = cs[0], cs[1], cs[2]

        f1 = self.fuser1(torch.cat([t1, c1], dim=1))  #[bv, 256, 32, 32]
        f1 = self.upsampler1(f1)                      #[bv, 128, 64, 64]

        f2 = self.fuser2(torch.cat([t2, c2], dim=1))  #[bv, 128, 64, 64]
        f2 = self.upsampler2(f1 + f2)                 #[bv, 64, 128, 128]

        f3 = self.fuser3(torch.cat([t3, c3], dim=1))  #[bv, 64, 128, 128]
        f3 = self.upsampler3(f2 + f3)                 #[bv, 64, 256, 256]

        return f3      

    
    def correlation_fpn_fusion(self, x_list):
        """
            (vb) num_depth_can h w: /8 /4 /2
        """
        x1, x2, x3 = x_list[0], x_list[1], x_list[2]
        x1 = self.layer1_outconv(x1)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x1 + x2)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        x3 = self.layer3_outconv(x3)
        x3 = self.layer2_outconv2(x2 + x3)
        
        return x3                                    #[bv, 128, 128, 128]

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        batch=None
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        """
            features  ([b, v, 256, 32, 32]), ([b, v, 128, 64, 64]), ([b, v, 64, 128, 128])
            intrinsics  torch.Size([b, v, 3, 3]): normalized
            extrinsics  torch.Size([b, v, 4, 4])
            near  torch.Size([b, v])
            far  torch.Size([b, v])

            cnn_features ([b, v, 256, 32, 32]), ([b, v, 128, 64, 64]), ([b, v, 64, 128, 128])
        """

        depth_mkpt_list, near_mks, far_mks, pc_mkpt_world_list = estimate_depth_from_mkpt(batch["context"]["image"], intrinsics, extrinsics, batch, True)
        print("--------------- gt near far")
        print(near, far)

        raw_correlation_list = []
        feat01_list = []
        for i in range(len(features)):
            # format the input
            b, v, c, h, w = features[i].shape
            feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
                # TODO: GT pose for now, Pose need to be estimated in prepare_feat_proj_data_lists
                prepare_feat_proj_data_lists( 
                    features[i],
                    intrinsics,
                    extrinsics,
                    near,
                    far,
                    num_samples=self.num_depth_candidates,
                    near_mk=near_mks, far_mk=far_mks, depth_guided_sample=False
                )
            ) # [(b v) d_candi 1, 1]

            if cnn_features is not None:
                cnn_features[i] = rearrange(cnn_features[i], "b v ... -> (v b) ...")

            # cost volume constructions
            feat01 = feat_comb_lists[0]
            feat01_list.append(feat01)

            if self.wo_cost_volume:
                raw_correlation_in = feat01
            else:
                raw_correlation_in_lists = []
                for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                    # sample feat01 from feat10 via camera projection
                    feat01_warped = warp_with_pose_depth_candidates(
                        feat10,
                        intr_curr,
                        pose_curr,
                        1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros",
                    )  # [B, C, D, H, W]
                    # calculate similarity
                    raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                        1
                    ) / (
                        c**0.5
                    )  # [vB, D, H, W]
                    raw_correlation_in_lists.append(raw_correlation_in)

                # average all cost volumes
                raw_correlation_in = torch.mean(
                    torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
                )  # [vxb d, h, w]
                raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

            # refine cost volume via 2D u-net
            if self.wo_cost_volume_refine:
                corr_project = getattr(self, "corr_project" + str(i))
                raw_correlation = corr_project(raw_correlation_in)
            else:
                corr_refine_net = getattr(self, "corr_refine_net" + str(i))
                raw_correlation = corr_refine_net(raw_correlation_in)  # (vb d_candi h w) 
                
                # apply skip connection
                regressor_residual = getattr(self, "regressor_residual" + str(i))
                raw_correlation = raw_correlation + regressor_residual(
                    raw_correlation_in
                ) # (vb d_candi h w) 

                raw_correlation_list.append(raw_correlation)

        # Perform raw correlation fpn fusion
        raw_correlation = self.correlation_fpn_fusion(raw_correlation_list)


        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [2xB, d_candi, 128, 128]
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, 128, 128)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax (vb, 1, 128, 128)
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor[-1]) # (vb, 1, 256, 256)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor[-1],
            mode="bilinear",
            align_corners=True,
        )# (vb, 1, 256, 256)


        # depth refinement
        # proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1)) # (vb, 128, 256, 256)
        proj_feat_in_fullres = self.trans_cnn_fpn_fusion(feat01_list, cnn_features) # (vb, 64, 256, 256)
        proj_feature = self.proj_feature(proj_feat_in_fullres) # (vb, 64, 256, 256)

        refine_out = self.refine_unet(torch.cat( 
            (extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1)) # (vb, 64, 256, 256)

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1) # (vb, 64 + 64 + 3, 256, 256)
        raw_gaussians = self.to_gaussians(raw_gaussians_in) # (vb, 84, 256, 256)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        ) # (b, v, 65536, 84)

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
        else:
            # delta fine depth and density
            delta_disps_density = self.to_disparity(refine_out) # (vb, 2, 256, 256)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            ) # (vb, 1, 256, 256)

            # combine coarse and fine info and match shape
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            ) # (b, v, 65536, 1, 1)

            fine_disps = (fullres_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            ) #(bv, 1, 256, 256)
            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            ) # (b, v, 65536, 1, 1)

            # pc_mkpt_world_list
            
            # in world coordiante
            def iproj_depth(depth, intr, extr):
                pc = iproj_full_img_(depth, intr.unsqueeze(0), extr.unsqueeze(0)).squeeze(0)
                perm = torch.randperm(pc.size(0))
                idx = perm[:1000]
                pc = pc[idx]
                return pc

            h, w = batch["context"]["image"].shape[-2:]
            for i in range(depths.shape[0]):
                mkpt_pc = pc_mkpt_world_list[i]
                
                depth_pc0 = iproj_depth(
                    rearrange(depths[i][0].squeeze(), "(h w) -> h w", h=h, w=w),
                    batch["context"]["intrinsics"][i][0],
                    batch["context"]["extrinsics"][i][0])
                
                depth_pc1 = iproj_depth(
                    rearrange(depths[i][1].squeeze(), "(h w) -> h w", h=h, w=w),
                    batch["context"]["intrinsics"][i][1],
                    batch["context"]["extrinsics"][i][1])
                
                colors = torch.zeros((mkpt_pc.shape[0] + depth_pc0.shape[0], 3))
                colors[:mkpt_pc.shape[0], 0] = 255
                colors[mkpt_pc.shape[0]:, 1] = 255

                colors2 = torch.zeros((depth_pc0.shape[0] + depth_pc1.shape[0], 3))
                colors2[:depth_pc0.shape[0], 0] = 255
                colors2[depth_pc0.shape[0]:, 1] = 255

                save_color_points_ply(torch.cat([depth_pc0, depth_pc1], dim=0), colors2, f"outputs/tmp/depths{i}.ply")
                save_color_points_ply(torch.cat([mkpt_pc, depth_pc0], dim=0), colors, f"outputs/tmp/depth_tri0_b{i}.ply")
                save_color_points_ply(torch.cat([mkpt_pc, depth_pc1], dim=0), colors, f"outputs/tmp/depth_tri1_b{i}.ply")



        return depths, densities, raw_gaussians
