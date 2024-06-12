# Built-in imports
from typing import List, Optional, Union
from ufl.tensors import ListTensor

# Third-party imports
import numpy as np
from fenics import *

def f2n(
    var: Union[Function, List[Function], ListTensor], 
    W: Optional[FunctionSpace] = None,
) -> np.ndarray:
    """
    Fenics to Numpy
    Returns a numpy array containing fenics function values
    """
    if type(var) == list:
        return np.stack([f2n(v) for v in var])
    elif type(var) == ListTensor:
        return np.stack([f2n(project(v)) for v in var])

    if W is not None:
        d2v_map = dof_to_vertex_map(W)
        n_sub = W.dofmap().num_entity_dofs(0)

    fs = var.function_space()
    dof_maps = _dof_maps(fs)
    n_sub = fs.dofmap().num_entity_dofs(0)
    if n_sub > 1:
        d2v_map = dof_to_vertex_map(fs)

    vec = var.vector().get_local()
    arr = np.zeros_like(dof_maps, dtype=np.float64)
    for i in np.ndindex(dof_maps.shape):
        dmi = dof_maps[i]

        if np.isnan(dmi):
            continue

        dmi = int(dmi)

        if n_sub > 1:
            vmi = int(d2v_map[dmi]) // n_sub
            i = (i[0], vmi)

        arr[i] = vec[dmi]

    return arr

def v2f(
    val: Union[np.ndarray, Expression, Function],
    var: Optional[Function] = None,
    fs: Optional[FunctionSpace] = None,
    name: Optional[str] = None,
) -> Function:
    """
    Value (mixed) to Fenics
    Set a value to a new or existing fenics variable.
    """
    assert var is not None or fs is not None

    if var is None:
        var = Function(fs, name=name)

        # d2v_map = dof_to_vertex_map(fs)
        # n_sub = fs.dofmap().num_entity_dofs(0)

    # If numpy array passed, set these as the function values
    if isinstance(val, np.ndarray):
        _set_vals_from_numpy(var, val)

    # If an expression is passed, interpolate on the space
    elif isinstance(val, Expr):
        var.assign(interpolate(val, var.function_space()))

    # If a function is passed, just assign
    elif isinstance(val, Function):
        var.assign(val)

    # otherwise raise an error
    else:
        raise RuntimeError("Unknown value type to convert")

    return var

def _set_vals_from_numpy(var: Function, values: np.ndarray):
    """
    Sets the vertex-values (or between-vertex-values) of a variable from a numpy array
    """
    fs = var.function_space()
    dof_maps = _dof_maps(fs)
    n_sub = fs.dofmap().num_entity_dofs(0)
    if n_sub > 1:
        d2v_map = dof_to_vertex_map(fs)

    assert (
        values.shape == dof_maps.shape
    ), f"shapes don't match!  values: {values.shape}. dof_maps: {dof_maps.shape}"
    vec = var.vector()

    for i in np.ndindex(dof_maps.shape):
        dmi = dof_maps[i]

        if np.isnan(dmi):
            continue

        dmi = int(dmi)

        if n_sub > 1:
            vmi = int(d2v_map[dmi]) // n_sub
            i = (i[0], vmi)

        vec[dmi] = values[i]


def _dof_maps(fs: FunctionSpace) -> np.ndarray:
    """
    Returns a numpy array for the dof maps of the function space
    """
    n_sub = fs.num_sub_spaces()
    if n_sub > 0:
        dms = []
        shape1 = 1
        for d in range(n_sub):
            dmd = _dof_maps(fs.sub(d))
            shape1 = max(shape1, len(dmd))
            dms.append(dmd)
        shape = (len(dms), shape1)
        dof_map = np.ones(shape) * np.nan
        for i, v in enumerate(dms):
            dof_map[i, : len(v)] = v
    else:
        dof_map = np.array(fs.dofmap().dofs())

    return dof_map
