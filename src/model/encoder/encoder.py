from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")

# 泛型类（Generic Classes）
class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    # 抽象基类只能继承而不能实例化，子类要实例化必须先实现该方法。
    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> Gaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
