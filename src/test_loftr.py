import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
from torchvision.utils import save_image
from PIL import Image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.model.encoder.loftr.utils.full_config import full_default_cfg
    from src.model.encoder.loftr.utils.opt_config import opt_default_cfg
    from src.model.encoder.encoder_eloftr import EncoderELoFTR, reparameter
    from src.model.decoder import Decoder, DecoderSplattingCUDA

    from src.model.encoder.visualization.encoder_visualizer_eloftr import EncoderVisualizerELoFTR
    from src.visualization.vis_depth import viz_depth_tensor, vis_disparity





model_type = 'full' 
precision = 'fp32' 
if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)
    
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_1",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)
    h, w = batch["target"]["image"].shape[-2:]


    # load model
    encoder = EncoderELoFTR(cfg=cfg.model.encoder, backbone_cfg =_default_cfg )
    encoder = encoder#.cuda().eval()

    # encoder_visualizer = EncoderVisualizerELoFTR(cfg.model.encoder.visualizer, encoder)

    decoder = DecoderSplattingCUDA(cfg=cfg.model.decoder, dataset_cfg=cfg.dataset)
    decoder = decoder#.cuda().eval()
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)
    # for k in batch["context"].keys():
    #     batch["context"][k] = batch["context"][k].cuda()
    # for k in batch["target"].keys():
    #     batch["target"][k] = batch["target"][k].cuda()

    print ("context:", batch["context"]["image"].shape)
    print ("target:", batch["target"]["image"].shape)

    # depth_ckpt_path = "checkpoints/depth_predictor.ckpt"
    # depth_ckpt_path = "outputs/tmp/checkpoints/epoch_65-step_2500.ckpt"
    # print("==> Load depth_predictor checkpoint: %s" % depth_ckpt_path)
    # encoder.load_state_dict(torch.load(depth_ckpt_path), strict=False) # only load weight of depth_predictor

    # https://github.com/ccl-1/matchsplat/raw/main/checkpoints/depth_predictor.ckpt

    visualization_dump = {}
    gaussians = encoder(
        batch,#["context"], 
        0, 
        False, 
        visualization_dump=visualization_dump,
        scene_names=batch["scene"])

    input()

    # vis_results = encoder_visualizer.visualize(batch["context"], 0)
    # vis_gaussians = vis_results['gaussians'] #3 vis_height vis_width
    # vis_depth = vis_results['depth'] #3 vis_width vis_height
    # print ("vis_depth", vis_depth.shape)

    # save_image(vis_gaussians, cfg.output_dir + "gaussians.png")
    # save_image(vis_depth, cfg.output_dir + "raw_depth.png")

    save_image(batch["context"]["image"][0, 0], f"outputs/tmp/input_0.png")
    save_image(batch["context"]["image"][0, 1], f"outputs/tmp/input_1.png")

     # save encoder depth map
    depth_vis = (
        (visualization_dump["depth"].squeeze(-1).squeeze(-1)).cpu().detach()
    )
    for v_idx in range(depth_vis.shape[1]):
        vis_depth = viz_depth_tensor(1.0 / depth_vis[0, v_idx], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"outputs/tmp/depth_{v_idx}.png")

    output = decoder.forward(
        gaussians, 
        batch["target"]["extrinsics"],
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
        depth_mode=cfg.train.depth_mode,
    )


    rendered_color = output.color.squeeze(0)
    rendered_depth = output.depth.squeeze(0)
    print (rendered_depth.shape)
    print (rendered_color.shape)

    for i in range(rendered_color.shape[0]):
        # rendered_depth = torch.from_numpy(vis_disparity(rendered_depth[i].detach().cpu().numpy()))

        save_image(rendered_color[i], cfg.output_dir + f"rendered_color{i}.png")
        # save_image(rendered_depth[i], cfg.output_dir + f"rendered_depth{i}.png")


run()

