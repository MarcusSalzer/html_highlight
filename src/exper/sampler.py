import itertools
import random
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def generate(self) -> list[dict]:
        """Return list of parameter dicts."""
        pass


class GridSampler(Sampler):
    def __init__(self, grid: dict[str, list]):
        self.grid = grid

    def generate(self) -> list[dict]:
        keys = list(self.grid.keys())
        values = list(self.grid.values())

        combos = itertools.product(*values)

        return [dict(zip(keys, combo)) for combo in combos]


class RandomSampler(Sampler):
    def __init__(
        self,
        space: dict[str, tuple[int, int] | tuple[float, float]],
        n_samples: int,
        seed: int | None = None,
    ):
        self.space = space
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def generate(self) -> list[dict]:
        samples = []

        for _ in range(self.n_samples):
            sample = {}
            for key, (low, high) in self.space.items():
                if isinstance(low, int) and isinstance(high, int):
                    sample[key] = self.rng.randint(low, high)
                else:
                    sample[key] = self.rng.uniform(low, high)
            samples.append(sample)

        return samples
