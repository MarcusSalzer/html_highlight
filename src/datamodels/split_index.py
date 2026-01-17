import datetime

import pydantic


class SplitIndex(pydantic.BaseModel):
    date: datetime.date
    group_counts: dict[str, int]
    split_ratios: dict[str, float]

    # precomputed overlaps
    overlap: list[tuple[tuple[str, str], float]]
    overlap_ngram: int

    # map each example to its group
    id_to_group: dict[str, str]
