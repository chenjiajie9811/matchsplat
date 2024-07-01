import torch
import torch.nn as nn
import warnings
import hydra
import numpy as np
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from PIL import Image
from torchvision.utils import save_image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.model.encoder.encoder_gmcopo import EncoderGmCoPo
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
    config_name="main_gmcopo",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)
    # batch = next(dataset, 0)
    b, v, c, h, w = batch["target"]["image"].shape

    # load model
    encoder = EncoderGmCoPo(cfg.model.encoder)
    encoder = encoder.eval().to(device)
    print("Initialized encoder!")

    decoder = DecoderSplattingCUDA(cfg=cfg.model.decoder, dataset_cfg=cfg.dataset)
    decoder = decoder.eval().cuda()
    
    # input data to device
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)
    for k in batch["context"].keys():
        batch["context"][k] = batch["context"][k].cuda()
    for k in batch["target"].keys():
        batch["target"][k] = batch["target"][k].cuda()

    dummy_model = ModelDummyWrapper(encoder, decoder)

    ckpt_path = "outputs/gmcopo/checkpoints/epoch_524-step_10500.ckpt"
    print("==> Load depth_predictor checkpoint: %s" % ckpt_path)
    dummy_model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    # run model
    print("Start running model")
    gaussians = encoder(batch, 0, False, scene_names=batch["scene"])

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
    for i in range(rendered_color.shape[0]):
        save_image(rendered_color[i], f"outputs/tmp/rendered_color{i}.png")

    print("finished!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    run()