"""Explicit SGD row migration for matched candidate-head continuations.

Not an exact resume: replacing unknown rows necessarily changes their state.
Unchanged parameters and known rows retain saved momentum; all replaced unknown
rows get zero momentum, identically in structure-off and structure-on arms.
"""
import copy
import random
import numpy as np
import torch


def restore_sgd(optimizer, saved, old_signature, named_parameters,
                replaced_head=None, known_rows=None):
    """Signature is captured from the pinned original model BEFORE head replacement.

Reject changed parameter order, groups, unknown state kinds and mismatched shapes.
Use only SGD momentum, never reinterpret Adam or remap unnamed arbitrary models.
"""
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError('Only SGD migration is supported')
    parameters = list(named_parameters)
    if [name for name, _ in parameters] != [name for name, _ in old_signature]:
        raise ValueError('Parameter names/order changed')
    state = copy.deepcopy(saved)
    old_groups = state['param_groups']
    new_groups = optimizer.param_groups
    if len(old_groups) != len(new_groups):
        raise ValueError('Optimizer group count changed')
    ids = [key for group in old_groups for key in group['params']]
    actual = [parameter for group in new_groups for parameter in group['params']]
    if len(ids) != len(parameters) or len(actual) != len(parameters):
        raise ValueError('Optimizer parameter count changed')
    if len(set(ids)) != len(ids) or any(a is not p for a, (_, p) in zip(actual, parameters)):
        raise ValueError('Optimizer registration/order mismatch')
    if any(len(old['params']) != len(new['params']) for old, new in zip(old_groups, new_groups)):
        raise ValueError('Optimizer group partition changed')
    if replaced_head is not None:
        if replaced_head not in [name for name, _ in parameters] or not isinstance(known_rows, int) or known_rows < 1:
            raise ValueError('Invalid replaced head/known rows')
    migrated = []
    for key, (name, parameter), (_, shape) in zip(ids, parameters, old_signature):
        shape = tuple(shape)
        replace = name == replaced_head
        if replace:
            if (len(shape) != 2 or parameter.ndim != 2 or shape[1] != parameter.shape[1]
                    or known_rows > min(shape[0], parameter.shape[0])):
                raise ValueError('Invalid head row expansion')
        elif shape != tuple(parameter.shape):
            raise ValueError(f'Unchanged parameter shape mismatch: {name}')
        for kind, value in state['state'].get(key, {}).items():
            if kind != 'momentum_buffer' or not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f'Unexpected SGD state: {name}/{kind}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite SGD state: {name}')
            if replace:
                buffer = torch.zeros(tuple(parameter.shape), dtype=value.dtype, device=value.device)
                buffer[:known_rows].copy_(value[:known_rows])
                state['state'][key][kind] = buffer
                migrated.append(name)
    optimizer.load_state_dict(state)
    for name, parameter in parameters:
        for value in optimizer.state.get(parameter, {}).values():
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise RuntimeError('Loaded state validation failed')
        if name == replaced_head:
            buffer = optimizer.state.get(parameter, {}).get('momentum_buffer')
            if buffer is not None and torch.count_nonzero(buffer[known_rows:]):
                raise RuntimeError('Unknown-row momentum was not reset')
    return dict(replaced_head=replaced_head, known_rows=known_rows,
                momentum_migrated=migrated, optimizer_steps_executed=0,
                exact_uninterrupted_resume=False if replaced_head else None)


def restore_rng(checkpoint):
    """Restore after constructors/resize, BEFORE sampler/augmentation creation.

This only restores RNG fields; it is not proof of loader or trajectory replay.
"""
    random.setstate(checkpoint['rng_python'])
    np.random.set_state(checkpoint['rng_numpy'])
    torch.set_rng_state(checkpoint['rng_torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['rng_cuda']])
