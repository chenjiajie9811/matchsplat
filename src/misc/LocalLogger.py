import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        os.system(f"rm -r {LOG_PATH}")
    # @property装饰器会将方法转换为相同名称的只读属性。
    # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加（）
    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0
    

    # 在分布式训练中，如果有一些日志或者测试进程只应该在 RANK=0 中调用，可以考虑将相关的代码放入一个函数，
    # 同时该函数使用 @rank_zero_only 进行修
    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
