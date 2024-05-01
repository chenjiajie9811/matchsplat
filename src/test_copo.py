import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker

    from src.model.encoder.encoder_copo import EncoderCoPo
    
    
    from src.model.decoder import Decoder, DecoderSplattingCUDA

    from src.visualization.vis_depth import viz_depth_tensor, vis_disparity

    from src.evaluation.evalulation_copo import copo_summary


# device = "cpu"
device = "cuda"

class ModelDummyWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        pass




@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_copo",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    
    
    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())


    dataset = iter(data_module.train_dataloader())

    batch = next(dataset, 0)
    batch = next(dataset, 0)

    b, v, c, h, w = batch["target"]["image"].shape #[-2:]


    # load model
    encoder = EncoderCoPo(cfg=cfg.model.encoder)
    encoder = encoder.eval().to(device)

    # encoder_visualizer = EncoderVisualizerELoFTR(cfg.model.encoder.visualizer, encoder)

    decoder = DecoderSplattingCUDA(cfg=cfg.model.decoder, dataset_cfg=cfg.dataset)
    decoder = decoder.eval().cuda()
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)
    for k in batch["context"].keys():
        batch["context"][k] = batch["context"][k].cuda()
    for k in batch["target"].keys():
        batch["target"][k] = batch["target"][k].cuda()

    dummy_model = ModelDummyWrapper(encoder, decoder)
    
    # depth_ckpt_path = "checkpoints/depth_predictor.ckpt"
    ckpt_path = "outputs/tmp/checkpoints/epoch_149-step_3000.ckpt"
    print("==> Load depth_predictor checkpoint: %s" % ckpt_path)
    dummy_model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    # https://github.com/ccl-1/matchsplat/raw/main/checkpoints/depth_predictor.ckpt

    visualization_dump = {}
    gaussians = encoder(
        batch, 
        0, 
        False, 
        visualization_dump=visualization_dump,
        scene_names=batch["scene"])

    
    for i in range(b):
        save_image(batch["context"]["image"][i, 0], f"outputs/tmp/input_{i}_0.png")
        save_image(batch["context"]["image"][i, 1], f"outputs/tmp/input_{i}_1.png")

     # save encoder depth map
    depth_vis = (
        (visualization_dump["depth"].squeeze(-1).squeeze(-1)).cpu().detach()
    )
    for b_idx in range(depth_vis.shape[0]):
        vis_depth = viz_depth_tensor(1.0 / depth_vis[b_idx, 0], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"outputs/tmp/depth_{b_idx}_0.png")

        vis_depth = viz_depth_tensor(1.0 / depth_vis[b_idx, 1], return_numpy=True)  # inverse depth
        Image.fromarray(vis_depth).save(f"outputs/tmp/depth_{b_idx}_1.png")

    
    

    copo_ret = copo_summary(batch)
    def save_numpy(img, path):
        im = Image.fromarray(img.astype(np.uint8))
        im.save(path)
    for i in range(b):
        save_image(copo_ret["warped_img"][i] / 255, f"outputs/tmp/warped_img_{i}.png")
        save_image(copo_ret["masked_img"][i], f"outputs/tmp/masked_img{i}.png")
        save_numpy(copo_ret["epipolar_pred"][i], f"outputs/tmp/epipolar_pred{i}.png")
        save_numpy(copo_ret["epipolar_gt"][i], f"outputs/tmp/epipolar_gt{i}.png")
        save_numpy(copo_ret["warped_img_mask"][i], f"outputs/tmp/warped_img_mask{i}.png")

    print ("saved copo!")
    # input()

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
    # print (rendered_depth.shape)
    # print (rendered_color.shape)

    for i in range(rendered_color.shape[0]):
        # rendered_depth = torch.from_numpy(vis_disparity(rendered_depth[i].detach().cpu().numpy()))

        save_image(rendered_color[i], cfg.output_dir + f"rendered_color{i}.png")
        # save_image(rendered_depth[i], cfg.output_dir + f"rendered_depth{i}.png")


run()

