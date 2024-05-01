import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.transforms as tf
from einops.einops import rearrange
from jaxtyping import Float
from collections import OrderedDict

from dataclasses import dataclass
from typing import Literal, Optional, List

# copo
from .copo.aggregation import UFC
from .copo.backbone import CrossBlock, SpatialEncoder
from .copo.utils import r6d2mat, extract_intrinsics, normalize_imagenet, warp, pose_inverse_4x4

# dataset
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians


from .encoder import Encoder
from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.get_depth import DepthPredictorMultiView

# from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from .visualization.encoder_visualizer_eloftr_cfg import EncoderVisualizerELoftrCfg

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCoPoCfg:
    name: Literal["copo"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerELoftrCfg

    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int

    copo_weights_path: str | None
    downscale_factor: int
    n_view: int
    n_points: int
    num_hidden_units_phi: int

    shim_patch_size: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool


class EncoderCoPo(Encoder[EncoderCoPoCfg]):
    backbone: UFC
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCoPoCfg, backbone_cfg=None) -> None:
        super().__init__(cfg)
        # self.config = backbone_cfg
        self.return_cnn_features = True

        self.backbone = UFC

        ckpt_path = cfg.copo_weights_path
        # if get_cfg().mode == 'train':
        #     if cfg.copo_weights_path is None:
        #         print("==> Init E-loFTR backbone from scratch")
        #     else:
        #         print("==> Load E-loFTR backbone checkpoint: %s" % ckpt_path)
        #         self.matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
                # self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
                # if precision == 'fp16':
                #     encoder = self.matcher.half()

        # ---------------------------------------------------------------
        # pose estimation

        self.cross_attention = CrossBlock()
        self.pose_regressor = nn.Sequential(
            nn.Linear((16*16+6) *256 * 2, 512 ),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 2),
            nn.ReLU(),
        )
        self.rotation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.translation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        # ---------------------------------------------------------------
        # Feature and cost aggregation

        self.feature_cost_aggregation = UFC()
        self.encoder = SpatialEncoder(use_first_pool=False, num_layers=5)
        self.latent_dim = 256*3 + 64

        self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)


        

        self.conv_1x1 = nn.Conv2d(256, cfg.d_feature, kernel_size=1)
        
        # ---------------------------------------------------------------

        
        
        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # TODO BA based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim, # input channels

            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )



    def get_z(self, context, val=False):
        """
        Extract features, estimate pose and find correspondence fields.
        
        Args:
        input (dict): A dictionary containing the input data.
        Returns:
        tuple: extracted features, estimated pose, correspondence fields.
        """
      
        rgb = context['image']
        B, n_ctxt, C, H, W= rgb.shape

        intrinsics = context['intrinsics']
      
        # Flatten first two dims (batch and number of context)
        rgb = torch.flatten(rgb, 0, 1)
        intrinsics = torch.flatten(intrinsics, 0, 1)
        intrinsics = intrinsics[:, None, :, :]
        # rgb = rgb.permute(0, -1, 1, 2) # (b*n_ctxt, ch, H, W)
        self.H, self.W = H, W
        
        rgb = (rgb + 1) / 2.
        rgb = normalize_imagenet(rgb)
        rgb = torch.cat([rgb], dim=1)
      
        z_cnn = self.encoder.forward(rgb, None, self.cfg.n_view)[:3] # (b*n_ctxt, self.latent_dim, H, W)
        # z_conv = self.conv_map(rgb[:(B*n_ctxt)])
        
        z_ctxts, flow_ctxts, c_ctxts = self.feature_cost_aggregation(z_cnn, self.cfg.n_view) # context 2 to context 1 flow and feature maps
        
        # Normalize intrinsics for a 0-1 image
        intrinsics_norm = context['intrinsics'].clone()
        intrinsics_norm[:, :, :2, :] = intrinsics_norm[:, :, :2, :] / self.H
        extracted_intrinsics = extract_intrinsics(intrinsics_norm)
        
        pose_feat_ctxt = self.cross_attention(
            z_ctxts[-1].flatten(-2,-1).transpose(-1,-2), 
            corr=c_ctxts, 
            intrinsics=extracted_intrinsics).reshape([B,-1]) # ctxt 1 and ctxt 2
        
        # z_ctxts = z_ctxts + [z_conv]
       
        pose_latent_ctxt = self.pose_regressor(pose_feat_ctxt)[:,:128]
        rot_ctxt = self.rotation_regressor(pose_latent_ctxt) # Bxn_views x 9,
        tran_ctxt = self.translation_regressor(pose_latent_ctxt) # Bxn_views x 3 
        R_ctxt = r6d2mat(rot_ctxt)[:, :3, :3] 

        #estimated pose between query and context 2?
        # Or context 1 and 2?
        estimated_rel_pose_ctxt = torch.cat(
            (torch.cat((R_ctxt, tran_ctxt.unsqueeze(-1)), dim=-1),
            torch.FloatTensor([0,0,0,1]).expand(B,1,-1).to(tran_ctxt.device)), 
            dim=1) 
        
        z_ctxts[-1] = F.relu(self.conv_1x1(z_ctxts[-1]))
        z_trans = [rearrange(z, "(b v) c h w -> b v c h w", b=B, v=n_ctxt) for z in z_ctxts]
        z_cnn = [rearrange(z, "(b v) c h w -> b v c h w",  b=B, v=n_ctxt) for z in z_cnn]

        return z_trans, z_cnn, estimated_rel_pose_ctxt, flow_ctxts

    def map_pdf_to_opacity(
            self,
            pdf: Float[Tensor, " *batch"],
            global_step: int,
        ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


    def forward(
        self,
        batch: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        # from CoPoNerf
        z=None, rel_pose=None, val=False, flow=None, debug=False
    ) -> Gaussians:
        """ 
        Update:
            batch (dict): {
                'flow': (torch.Tensor): 
                'rel_pose_flip': (torch.Tensor): 
                'rel_pose' : (torch.Tensor): 
                'gt_rel_pose' : (torch.Tensor): 
                'gt_rel_pose_flip' : 
            }
        """
        context = batch["context"]
        device = context["image"].device
        b, v, _, h, w = context["image"].shape      # 224, 320
        
        trans_features, cnn_features, estimated_rel_pose, flow_orig = self.get_z(context) 
        # estimated pose [0] == context 1 to context 2, [1] == context 2 to query?

        batch['flow'] = flow_orig
        batch['rel_pose_flip'] = pose_inverse_4x4(estimated_rel_pose)
        batch['rel_pose'] = (estimated_rel_pose)
        batch['gt_rel_pose'] = torch.matmul(torch.inverse(context['extrinsics'][:,0]), context['extrinsics'][:,1])
        batch['gt_rel_pose_flip'] = torch.inverse(torch.matmul(torch.inverse(context['extrinsics'][:,-1]), context['extrinsics'][:,0]))
       
        """
            I guess coponerf internelly assumes the num input views can only be 2

            trans features list(3)
            torch.Size([b, 2, 256, 16, 16]) torch.Size([b, 2, 256, 32, 32]) torch.Size([b, 2, 256, 64, 64]) and after the conv1x1 to [b, 2, 128, 64, 64]

            cnn features  list(3)
            torch.Size([b, 2, 512, 16, 16]) torch.Size([b, 2, 256, 32, 32]) torch.Size([b, 2, 128, 64, 64])

            estimated_rel_pose:
            torch.Size([b, 4, 4])

            flow: tuple(4)
            flow (2 -> 1), flow_flip (1 -> 2), flow_t_to_s, flow_s_to_t
            all with size (b, 2 64, 64)
        """
        

        
        # Sample depths from the resulting features.
        in_feats = trans_features[-1]

        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features[-1],
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
    
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None


