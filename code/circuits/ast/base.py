from __future__ import annotations

import re

from typing import NamedTuple, Any, Optional, TypeVar, Protocol, Type

__all__ = ['_check_and_raise_error']


version_re = re.compile(r'(\d+)\.(\d+)\.(\d+)([a-z]?)')


T_cov = TypeVar('T_cov', covariant=True)
T = TypeVar('T')
T2 = TypeVar('T2')


def _check_and_raise_error(val: Any, type_: Type[Any], val_name: str) -> None:
    if not isinstance(val, type_):
        raise ValueError(val_name + f" must be of type `{type_.__name__}` but it is "
                                    f"`{type(val).__name__}`")


class AST(Protocol[T_cov]):
    """
    All ast classes must comply with this specification/protocol. This is used when
    passing generic ast classes/objects down to other functions. See
    `_load_list_from_json_dict` for a usage example.
    """
    @classmethod
    def from_json_obj(cls, val: Any) -> T_cov:
        raise NotImplementedError


class Version(NamedTuple):
    major: int
    minor: int
    revision: int
    revision_minor: Optional[str] = None  # single alphanumeric character

    @classmethod
    def from_json_obj(cls, val: str) -> Version:
        _check_and_raise_error(val, str, "A token")

        match = version_re.fullmatch(val)
        if match is None:
            raise ValueError(f"The string `{val}` cannot be interpreted as X.X.Xz")
        rev_minor = match[4] if match[4] else None
        return Version(int(match[1]), int(match[2]), int(match[3]), rev_minor)

    @property
    def human_readable_version(self) -> str:
        if self.revision_minor:
            return f"{self.major}.{self.minor}.{self.revision}{self.revision_minor}"
        else:
            return f"{self.major}.{self.minor}.{self.revision}"
