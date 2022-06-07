import pathlib
import json

from typing import Union, Dict, Any

from .sncircuit import SNCircuit
from .ast.base import _check_and_raise_error, Version

__all__ = ['from_json_path', 'from_json_str', 'from_json_obj']


def from_json_path(
    path: Union[str, pathlib.Path],
    args: Dict[str, float]
) -> SNCircuit:
    with open(path) as f:
        data = f.read()
    return from_json_str(data, args)


def from_json_str(
    data: str,
    args: Dict[str, float]
) -> SNCircuit:
    json_obj = json.loads(data)
    return from_json_obj(json_obj, args)


def from_json_obj(
    data: Any,
    args: Dict[str, float]
) -> SNCircuit:
    _check_and_raise_error(data, dict, "The top JSON structure")
    if 'version' not in data:
        raise ValueError("Version missing from JSON")

    version = Version.from_json_obj(data['version'])
    if version == Version(0, 0, 1):
        from .ast.v_0_0_1.load import from_ast as from_ast_v1
        from .ast.v_0_0_1.ast import SNFile as SNFile_v1

        ast_circuit_v1 = SNFile_v1.from_json_obj(data)
        return from_ast_v1(ast_circuit_v1, args)
    else:
        raise ValueError(f"File version {version.human_readable_version} not supported")
