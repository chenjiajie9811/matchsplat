import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import torch

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

    # load model
    encoder = EncoderELoFTR(cfg=cfg.model.encoder, backbone_cfg =_default_cfg )
    encoder = encoder.eval()
    data_shim = get_data_shim(encoder)
    batch = data_shim(batch)

    encoder(batch["context"], 0, False, scene_names=batch["scene"])

run()

