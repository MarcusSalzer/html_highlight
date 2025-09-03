from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DatasetRecord:
    name: str
    lang: str
    tokens: list[str]
    tags: list[str]
    difficulty: str

    @property
    def id(self) -> str:
        """Id like `name_lang`"""
        return f"{self.name}_{self.lang}"

    def toDict(self, with_id: bool = False):
        d = asdict(self)
        if with_id:
            d["id"] = self.id
        return d
