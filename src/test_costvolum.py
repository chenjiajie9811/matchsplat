
import warnings
import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from src.model.encoder.encoder_costvolume import EncoderCostVolume

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # load data
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())
    batch = next(dataset, 0)

    # load model
    encoder = EncoderCostVolume(cfg.model.encoder)
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)

    # run model
    gaussians = encoder(batch["context"], 0, False, scene_names=batch["scene"])


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    run()