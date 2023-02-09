import jax
from jax._src.api import *
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, argnames_partial_except,
    validate_argnames, validate_argnums, check_callable, resolve_argnums,
    FLAGS)

del ad
import jaxy_ad


# def value_and_grad(f):
#   return jax.value_and_grad(f)


def value_and_grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False,
                   allow_int: bool = False, reduce_axes: Sequence[AxisName] = ()
  ) -> Callable[..., Tuple[Any, Any]]:
  """Create a function that evaluates both ``fun`` and the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``value_and_grad(f, reduce_axes=('batch',))`` will
      create a function that computes the total gradient while
      ``value_and_grad(f)`` will create one that computes the per-example
      gradient.

  Returns:
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a sequence of integers, the gradient is a tuple of values with the same
    shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a tuple of ((value, auxiliary_data), gradient) is returned.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  check_callable(fun)
  argnums = core.concrete_or_error(_ensure_index, argnums)
  reduce_axes = _ensure_str_tuple(reduce_axes)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to {argnums=} requires at least "
                      f"{max_argnum + 1} positional arguments to be passed by the caller, "
                      f"but got only {len(args)} positional arguments.")

    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    for leaf in tree_leaves(dyn_args):
      _check_input_dtype_grad(holomorphic, allow_int, leaf)
    if not has_aux:
      ans, vjp_py = _vjp(f_partial, *dyn_args, reduce_axes=reduce_axes)
    else:
      ans, vjp_py, aux = _vjp(
          f_partial, *dyn_args, has_aux=True, reduce_axes=reduce_axes)
    _check_scalar(ans)
    tree_map(partial(_check_output_dtype_grad, holomorphic), ans)
    g = vjp_py(lax_internal._one(ans))
    g = g[0] if isinstance(argnums, int) else g
    if not has_aux:
      return ans, g
    else:
      return (ans, aux), g

  return value_and_grad_f

@overload
def vjp(fun: Callable[..., T],
        *primals: Any,
        has_aux: Literal[False] = False,
        reduce_axes: Sequence[AxisName] = ()) -> Tuple[T, Callable]:
  ...

@overload
def vjp(fun: Callable[..., Tuple[T, U]], *primals: Any,
        has_aux: Literal[True],
        reduce_axes: Sequence[AxisName] = ()) -> Tuple[T, Callable, U]:
  ...
def vjp(  # type: ignore
    fun: Callable, *primals, has_aux: bool = False, reduce_axes=()
  ) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
  """Compute a (reverse-mode) vector-Jacobian product of ``fun``.

  :py:func:`grad` is implemented as a special case of :py:func:`vjp`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: A sequence of primal values at which the Jacobian of ``fun``
      should be evaluated. The number of ``primals`` should be equal to the
      number of positional parameters of ``fun``. Each primal value should be
      an array, a scalar, or a pytree (standard Python containers) thereof.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      VJP will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``vjp(f, *args, reduce_axes=('batch',))`` will
      create a VJP function that sums over the batch while ``vjp(f, *args)``
      will create a per-example VJP.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, vjpfun)`` pair, where
    ``primals_out`` is ``fun(*primals)``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, vjpfun, aux)`` tuple where ``aux`` is the auxiliary data
    returned by ``fun``.

    ``vjpfun`` is a function from a cotangent vector with the same shape as
    ``primals_out`` to a tuple of cotangent vectors with the same number and
    shapes as ``primals``, representing the vector-Jacobian product of ``fun``
    evaluated at ``primals``.

  >>> import jax
  >>>
  >>> def f(x, y):
  ...   return jax.numpy.sin(x), jax.numpy.cos(y)
  ...
  >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
  >>> xbar, ybar = f_vjp((-0.7, 0.3))
  >>> print(xbar)
  -0.61430776
  >>> print(ybar)
  -0.2524413
  """
  check_callable(fun)
  reduce_axes = _ensure_str_tuple(reduce_axes)
  return _vjp(
      lu.wrap_init(fun), *primals, has_aux=has_aux, reduce_axes=reduce_axes)

def _vjp(fun: lu.WrappedFun, *primals, has_aux=False, reduce_axes=()):
  """Variant of vjp() that takes an lu.WrappedFun."""
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: dispatch.check_arg(arg)
  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    out_primal, out_vjp = jaxy_ad.vjp(
        flat_fun, primals_flat, reduce_axes=reduce_axes)
    out_tree = out_tree()
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, in_tree)
    out_primal, out_vjp, aux = jaxy_ad.vjp(
        flat_fun, primals_flat, has_aux=True, reduce_axes=reduce_axes)
    out_tree, aux_tree = out_aux_trees()
  out_primal_py = tree_unflatten(out_tree, out_primal)
  ct_dtypes = [core.primal_dtype_to_tangent_dtype(_dtype(x)) for x in out_primal]
  ct_shapes = [np.shape(x) for x in out_primal]
  # Ensure that vjp_py is a PyTree so that we can pass it from the forward to the
  # backward pass in a custom VJP.
  vjp_py = Partial(partial(_vjp_pullback_wrapper, fun.__name__,
                           ct_dtypes, ct_shapes, (out_tree, in_tree)),
                   out_vjp)
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)

_dtype = partial(dtypes.dtype, canonicalize=True)

def _vjp_pullback_wrapper(name, cotangent_dtypes, cotangent_shapes, io_tree,
                          fun, *py_args_):
  if len(py_args_) != 1:
    msg = (f"The function returned by `jax.vjp` applied to {name} was called "
           f"with {len(py_args_)} arguments, but functions returned by "
           "`jax.vjp` must be called with a single argument corresponding to "
           f"the single value returned by {name} (even if that returned "
           "value is a tuple or other container).\n"
           "\n"
           "For example, if we have:\n"
           "\n"
           "  def f(x):\n"
           "    return (x, x)\n"
           "  _, f_vjp = jax.vjp(f, 1.0)\n"
           "\n"
           "the function `f` returns a single tuple as output, and so we call "
           "`f_vjp` with a single tuple as its argument:\n"
           "\n"
           "  x_bar, = f_vjp((2.0, 2.0))\n"
           "\n"
           "If we instead call `f_vjp(2.0, 2.0)`, with the values 'splatted "
           "out' as arguments rather than in a tuple, this error can arise.")
    raise TypeError(msg)
  py_args, = py_args_
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    raise TypeError(f"Tree structure of cotangent input {in_tree}, does not match structure of "
                    f"primal output {in_tree_expected}.")
  for arg, ct_dtype, ct_shape in safe_zip(args, cotangent_dtypes, cotangent_shapes):
    expected_tangent_dtype = core.primal_dtype_to_tangent_dtype(_dtype(arg))
    if expected_tangent_dtype != ct_dtype:
      raise TypeError(
          f"Type of cotangent input to vjp pullback function ({ct_dtype}) is not "
          f"the expected tangent type ({expected_tangent_dtype}) of corresponding primal output "
          f"with dtype {_dtype(arg)}.")
    if np.shape(arg) != ct_shape:
      raise ValueError(
          f"Shape of cotangent input to vjp pullback function {np.shape(arg)} "
          "must be the same as the shape of corresponding primal input "
          f"{ct_shape}.")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)



def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.get_aval(x)
  except TypeError as e:
    raise TypeError(msg(f"was {x}")) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        raise TypeError(msg(f"had shape: {aval.shape}"))
    else:
      raise TypeError(msg(f"had abstract value {aval}"))

def _check_input_dtype_revderiv(name, holomorphic, allow_int, x):
  dispatch.check_arg(x)
  aval = core.get_aval(x)
  if core.is_opaque_dtype(aval.dtype):
    raise TypeError(
        f"{name} with input element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires inputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  if (dtypes.issubdtype(aval.dtype, np.integer) or
      dtypes.issubdtype(aval.dtype, np.bool_)):
    if not allow_int:
      raise TypeError(f"{name} requires real- or complex-valued inputs (input dtype "
                      f"that is a sub-dtype of np.inexact), but got {aval.dtype.name}. "
                      "If you want to use Boolean- or integer-valued inputs, use vjp "
                      "or set allow_int to True.")
  elif not dtypes.issubdtype(aval.dtype, np.inexact):
    raise TypeError(f"{name} requires numerical-valued inputs (input dtype that is a "
                    f"sub-dtype of np.bool_ or np.number), but got {aval.dtype.name}.")
_check_input_dtype_grad = partial(_check_input_dtype_revderiv, "grad")

def _check_output_dtype_revderiv(name, holomorphic, x):
  aval = core.get_aval(x)
  if core.is_opaque_dtype(aval.dtype):
    raise TypeError(
        f"{name} with output element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  elif dtypes.issubdtype(aval.dtype, np.complexfloating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving complex "
                    "outputs, use jax.vjp directly.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For differentiation of functions with integer outputs, use "
                    "jax.vjp directly.")
_check_output_dtype_grad = partial(_check_output_dtype_revderiv, "grad")
