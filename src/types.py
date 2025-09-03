from typing import Any, Mapping
import pydantic
from typing import Iterator, Protocol, TypeVar


T = TypeVar("T", covariant=True)


class ArrayLike(Protocol[T]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __getitem__(self, index: int) -> T: ...


class LrsPlatConfig(pydantic.BaseModel):
    factor: float
    patience: int


class TrainConfig(pydantic.BaseModel):
    epochs: int
    bs: int
    lr: float
    lrs: Any | None = None
    lrs_plat: LrsPlatConfig | None = None
    stop_patience: int | None = None


class ModelTrainSetup(pydantic.BaseModel):
    """Complete config for training model"""

    split: str
    constructor: dict[str, int | float | bool]
    train: TrainConfig


class TrainResultMeta(pydantic.BaseModel):
    setup: ModelTrainSetup
    vocab: list[str]
    tag_vocab: list[str]
    metrics: Mapping[str, list[float]] | None = None
