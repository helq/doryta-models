from __future__ import annotations

import re

from typing import NamedTuple, List, Union, Dict, Any, Type

from ..base import _check_and_raise_error, AST, T, T2, Version


token_re = re.compile(r'[a-z][a-zA-Z0-9/_-]*')
param_re = re.compile(r'\w+')


class Token(NamedTuple):
    value: str

    @classmethod
    def from_json_obj(cls, val: str) -> Token:
        _check_and_raise_error(val, str, "A token")

        match = token_re.fullmatch(val)
        if match is None:
            raise ValueError(f"The string `{val}` cannot be interpreted as a token")
        return Token(match[0])


class ParamName(NamedTuple):
    value: str

    @classmethod
    def from_json_obj(cls, val: str) -> ParamName:
        _check_and_raise_error(val, str, "The name of a parameter")

        match = param_re.fullmatch(val)
        if match is None:
            raise ValueError(f"The string `{val}` cannot be interpreted as a parameter's name")
        return ParamName(match[0])


class ParamValue(NamedTuple):
    value: Union[int, float, Token]

    @classmethod
    def from_json_obj(cls, input: Any) -> ParamValue:
        if isinstance(input, str):
            value: Union[int, float, Token] = Token.from_json_obj(input)
        elif isinstance(input, (int, float)):
            value = input
        else:
            raise ValueError(f"The value `{input}` is not valid parameter value. Only"
                             "values: int, float and args allowed.")
        return ParamValue(value)


class SynapseParams(NamedTuple):
    params: Dict[ParamName, ParamValue]

    @classmethod
    def from_json_obj(cls, input: Dict[str, Any]) -> SynapseParams:
        _check_and_raise_error(input, dict, "Synapse params")

        params = {ParamName.from_json_obj(k): ParamValue.from_json_obj(v)
                  for k, v in input.items()}
        return SynapseParams(params)


def _load_list_from_json_dict(
    input: Dict[str, Any], key: str, func: Type[AST[T]], msg: str
) -> List[T]:
    """
    Given some input dictionary obtained from de-serializing JSON, this fuction creates a
    list of AST values (e.g, Tokens) from a list stored in `input[key]`.
    For example,

    ```python
        >>> inp = {'args': ['a', 'b'], 'output': ['that']}
        >>> _load_list_from_json_dict(inp, "args", Token, msg="args")
        [Token('a'), Token('b')]
        >>> _load_list_from_json_dict(inp, "output", Token, msg="output")
        [Token('that')]
    ```
    """
    if key in input:
        _check_and_raise_error(input[key], list, msg)
        return [func.from_json_obj(val) for val in input[key]]
    return []


def _load_dict_from_json_dict(
    input: Dict[str, Any],
    key: str,
    keyFunc: Type[AST[T]],
    func: Type[AST[T2]],
    msg: str
) -> Dict[T, T2]:
    if key in input:
        _check_and_raise_error(input[key], dict, msg)
        return {keyFunc.from_json_obj(name): func.from_json_obj(val)
                for name, val in input[key].items()}
    return {}


class NeuronDef(NamedTuple):
    synapses: Dict[Token, SynapseParams]
    params: Dict[ParamName, ParamValue]

    @classmethod
    def from_json_obj(cls, input: Dict[str, Any]) -> NeuronDef:
        _check_and_raise_error(input, dict, "Neuron's input")

        extreneous_keys = set(input.keys()) - {'synapses', 'params'}
        if extreneous_keys:
            raise ValueError(f"The keys `{extreneous_keys}` are invalid neuron arguments")

        return NeuronDef(
            synapses=_load_dict_from_json_dict(
                input, 'synapses', Token, SynapseParams, msg="`synapses` params for neuron"),
            params=_load_dict_from_json_dict(
                input, 'params', ParamName, ParamValue, msg="`params` params for neuron"))


class SNFile(NamedTuple):
    version: Version
    args: List[Token]
    params: Dict[ParamName, ParamValue]
    outputs: List[Token]
    inputs: List[Dict[Token, SynapseParams]]
    neurons: Dict[Token, NeuronDef]

    @classmethod
    def from_json_obj(self, inp: Dict[str, Any]) -> SNFile:
        _check_and_raise_error(inp, dict, "The top JSON structure")

        file_keys = {'version', 'args', 'params', 'outputs', 'inputs', 'neurons'}
        extreneous_keys = set(inp.keys()) - file_keys

        if extreneous_keys:
            raise ValueError(f"The keys `{extreneous_keys}` are invalid arguments "
                             "for the document.")

        if 'version' in inp and isinstance(inp['version'], str):
            version = Version.from_json_obj(inp['version'])
        else:
            raise ValueError("No version has been given or is not a string")

        inputs = []
        if 'inputs' in inp:
            _check_and_raise_error(inp['inputs'], list, "`inputs`")
            for input_dict in inp['inputs']:
                _check_and_raise_error(input_dict, dict, "All inputs in `input`")
                inputs.append({
                    Token.from_json_obj(s_name): SynapseParams.from_json_obj(synapse)
                    for s_name, synapse in input_dict.items()})

        return SNFile(
            version=version,
            args=_load_list_from_json_dict(inp, 'args', Token, msg="`args`"),
            params=_load_dict_from_json_dict(inp, 'params', ParamName, ParamValue, msg="`params`"),
            outputs=_load_list_from_json_dict(inp, 'outputs', Token, msg="`outputs`"),
            inputs=inputs,
            neurons=_load_dict_from_json_dict(inp, 'neurons', Token, NeuronDef, msg="`neurons`")
        )
