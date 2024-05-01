from typing import Optional
from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .encoder_eloftr import EncoderELoFTR, EncoderELoFTRCfg
from .encoder_copo import EncoderCoPo, EncoderCoPoCfg



from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume
from .visualization.encoder_visualizer_eloftr import EncoderVisualizerELoFTR

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "eloftr": (EncoderELoFTR, EncoderVisualizerCostVolume),
    "copo": (EncoderCoPo, EncoderVisualizerCostVolume),

}

# EncoderCfg = EncoderCostVolumeCfg
# EncoderCfg = EncoderELoFTRCfg
EncoderCfg = EncoderCostVolumeCfg | EncoderELoFTRCfg | EncoderCoPoCfg
 # 注意这个地方地 dataclass 和配置文件直接挂钩， 每次换配置文件之前 这里一定要对应好 ... 


def get_encoder(cfg: EncoderCfg, backbone_cfg =None) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg,backbone_cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer


# def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
#     encoder, visualizer = ENCODERS[cfg.name]
#     encoder = encoder(cfg)
#     if visualizer is not None:
#         visualizer = visualizer(cfg.visualizer, encoder)
#     return encoder, visualizer