from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from core.serialization import fingerprint

_MUTABLE_CONTAINERS = (dict, list, set, bytearray)


class FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__pydantic_init_subclass__(**kwargs)

        for name, field in cls.model_fields.items():
            if _contains_mutable(field.annotation):
                raise TypeError(f"{cls.__name__}.{name}: mutable container in a FrozenModel")

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def _contains_mutable(annotation: object) -> bool:
    origin = get_origin(annotation)

    if annotation in _MUTABLE_CONTAINERS or origin in _MUTABLE_CONTAINERS:
        return True
    if origin is not None or isinstance(annotation, UnionType) or origin is Union:
        return any(_contains_mutable(arg) for arg in get_args(annotation))
    return False
