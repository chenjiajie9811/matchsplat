import torch
from torch import Tensor, nn
import torchvision.transforms as tf
from einops.einops import rearrange
from jaxtyping import Float
from collections import OrderedDict

from dataclasses import dataclass
from typing import Literal, Optional, List

# loftr
from .loftr.loftr import LoFTR, reparameter

# dataset
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians


from .encoder import Encoder
from ...global_cfg import get_cfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.get_depth import DepthPredictorMultiView

from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderELoFTRCfg:
    name: Literal["eloftr"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg

    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int

    eloftr_weights_path: str | None
    downscale_factor: int

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


class EncoderELoFTR(Encoder[EncoderELoFTRCfg]):
    backbone: LoFTR
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderELoFTRCfg, backbone_cfg) -> None:
        super().__init__(cfg)
        self.config = backbone_cfg
        self.return_cnn_features = True

        self.matcher = LoFTR(backbone_cfg)   
        ckpt_path = cfg.eloftr_weights_path
        if get_cfg().mode == 'train':
            if cfg.eloftr_weights_path is None:
                print("==> Init E-loFTR backbone from scratch")
            else:
                print("==> Load E-loFTR backbone checkpoint: %s" % ckpt_path)
                self.matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
                self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
                # if precision == 'fp16':
                #     encoder = self.matcher.half()

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

    
    def data_process(self, images): 
        """  b v c h w -> b, 1, h, w,  range [0, 1]? """
        assert images.shape[1] == 2   # 2 VIEWS
        img0, img1 = images[:,0], images[:,1]

        to_gary = tf.Grayscale()
        img0_gray, img_gray = to_gary(img0), to_gary(img1)  # b 1 h w      
        data = {'image0': img0_gray, 'image1': img_gray}
        return data

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
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape      # 224, 320
        data = self.data_process(context["image"])  # input size must be divides by 32

        trans_features, cnn_features = self.matcher(data, self.return_cnn_features)  # Features are downsampled by 8 [28, 40]
        mkpts0, mkpts1, mconf = data['mkpts0_f'], data['mkpts1_f'], data['mconf']
        #  TODO : Depth need to be optimized by correspondence and BA ---------------------------------------------------------- TODO


        # Sample depths from the resulting features.
        in_feats = trans_features
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
            cnn_features=cnn_features,
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

        print("Testing Gaussians out: ", Gaussians)

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


