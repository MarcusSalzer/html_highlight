from dataclasses import dataclass


@dataclass(slots=True)
class OverlapStat:
    idxs: set[int]
    overlap: float
