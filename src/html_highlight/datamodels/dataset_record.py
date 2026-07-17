from typing import Any

from pydantic import BaseModel


class DatasetRecord(BaseModel):
    lang: str
    name: str
    tokens: list[str]
    tags: list[str]
    difficulty: str = "ambiguous"

    @property
    def id(self) -> str:
        """Id like `lang_name`"""
        return f"{self.lang}_{self.name}"

    def toDict(self, with_id: bool = False):
        d = self.model_dump()
        if with_id:
            d["id"] = self.id
        return d

    def to_string(self):
        """Simply join tokens"""
        return "".join(self.tokens)

    def model_post_init(self, context: Any) -> None:
        assert len(self.tokens) == len(self.tags), f"{len(self.tokens)=}, {len(self.tags)=}"
