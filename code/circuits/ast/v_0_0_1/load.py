import json

from typing import Dict, Union, Optional, Any
from collections.abc import Iterable

from .ast import SNFile, ParamName as AstParamName, ParamValue as AstParamValue, \
    Token as AstToken, SynapseParams as AstSynapseParams
from ..base import Version
from ...sncircuit import SNCircuit, SynapParams, Neuron, LIF


def from_json_str(
    data: str,
    args: Dict[str, float]
) -> SNCircuit:
    obj = json.loads(data)
    return from_json_obj(obj, args)


def from_json_obj(
    data: Any,
    args: Dict[str, float]
) -> SNCircuit:
    """
    Loading SNCircuit from AST
    """
    ast_circuit = SNFile.from_json_obj(data)
    return from_ast(ast_circuit, args)


def from_ast(
    ast: SNFile,
    args: Dict[str, float]
) -> SNCircuit:
    _raise_if_false(ast.version == Version(0, 0, 1),
                    f"Version {ast.version.human_readable_version} unsupported")
    _raise_if_false(sorted(arg.value for arg in ast.args) == sorted(args.keys()),
                    "Arguments given do not correspond to arguments needed")

    defaults_params = {
        'potential': 0.0,
        'current': 0.0,
        'resting_potential': 0.0,
        'reset_potential': 0.0,
    }
    g_params = _params_from_ast_params(args, ast.params, defaults_params)
    ids_to_int = {k.value: i for i, k in enumerate(ast.neurons.keys())}

    neurons = {}
    for neuron_tok, neuron_def in ast.neurons.items():
        n_params = _params_from_ast_params(
            args, neuron_def.params, g_params,
            keys=['resistance', 'capacitance', 'threshold', 'potential', 'current',
                  'resting_potential', 'reset_potential'],
            alias_keys={'R': 'resistance', 'C': 'capacitance'})

        neurons[ids_to_int[neuron_tok.value]] = Neuron(
            params=LIF(**n_params),
            synapses=_synapses_from_ast_synapses(
                neuron_def.synapses, args, g_params, ids_to_int)
        )

    return SNCircuit(
        outputs=[ids_to_int[o.value] for o in ast.outputs],
        inputs=[_synapses_from_ast_synapses(input, args, g_params, ids_to_int)
                for input in ast.inputs],
        inputs_id={},
        neurons=neurons,
        ids_to_int=ids_to_int)


def _raise_if_false(question: bool, msg: str) -> None:
    if not question:
        raise ValueError(msg)


def _params_from_ast_params(  # noqa: C901
    args:        Dict[str, Union[float, int]],
    params_dict: Dict[AstParamName, AstParamValue],
    defaults:    Optional[Dict[str, Union[float, int]]] = None,
    keys:        Optional[Iterable[str]] = None,
    alias_keys:  Optional[Dict[str, str]] = None
) -> Dict[str, Union[float, int]]:
    """
    This function takes a dictionary of AST parameters and returns a dictionary with no
    "gaps" (variable/argument names). All gaps are filled with the data given by `args`.
    The output dictionary can be initialized with a `defaults` input dict.
    The keys/parameters can be restricted to `keys`.
    `alias_keys` determines the aliases for all parameters
    """
    if defaults is None:
        params = {}
    elif keys is None:
        params = defaults.copy()
    else:
        if alias_keys is not None:
            keys = set(keys).union(alias_keys)
            # checking that final aliases should be contained within keys
            assert len(set(alias_keys.values()) - keys) == 0
        params = {k: defaults[k] for k in keys if k in defaults}

    # Removing converting all aliases into the same key
    if alias_keys is not None:
        for old_key, new_key in alias_keys.items():
            if old_key in params:
                assert new_key not in params
                params[new_key] = params.pop(old_key)

    for name, p_val in params_dict.items():
        # Parameter name
        p_name = name.value
        if alias_keys is not None and p_name in alias_keys:
            p_name = alias_keys[p_name]  # replacing key for alias

        if isinstance(p_val.value, AstToken):
            token_name = p_val.value.value
            if token_name not in args:
                raise ValueError(f"No argument `{p_val.value}` given")
            # The parameter's value is the one from args
            params[p_name] = args[token_name]
        else:
            # The parameter's value is the one from params
            params[p_name] = p_val.value

    return params


def _synapses_from_ast_synapses(
    ast_synapses: Dict[AstToken, AstSynapseParams],
    args: Dict[str, Union[float, int]],
    g_params: Dict[str, Union[float, int]],
    ids_to_int: Dict[str, int],
) -> Dict[int, SynapParams]:
    synapses = {}
    for n, s in ast_synapses.items():
        # n: neuron that synapse points to (Token)
        # s: synapse parameters (SynapseParams)
        s_params = _params_from_ast_params(
            args, s.params, g_params, keys=['weight', 'delay'])
        assert type(s_params['delay']) is int
        synapses[ids_to_int[n.value]] = SynapParams(**s_params)  # type: ignore
    return synapses
