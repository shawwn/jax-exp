# from jax.interpreters.ad import *
from jax.interpreters.ad import Zero, instantiate_zeros, zeros_like_aval, partial, Partial, UndefinedPrimal, linearize, \
  Literal, core, add_tangents, is_undefined_primal, Dict, Any, contextlib, source_info_util, reducing_transposes, \
  get_primitive_transpose, map, zip

import jax

def vjp(traceable, primals, has_aux=False, reduce_axes=()):
  if not has_aux:
    out_primals, pvals, jaxpr, consts = linearize(traceable, *primals)
  else:
    out_primals, pvals, jaxpr, consts, aux = linearize(traceable, *primals, has_aux=True)

  def unbound_vjp(pvals, jaxpr, consts, *cts):
    cts = tuple(ct for ct, pval in zip(cts, pvals) if not pval.is_known())
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = backward_pass(jaxpr, reduce_axes, True, consts, dummy_args, cts)
    return map(instantiate_zeros, arg_cts)

  # Ensure that vjp_ is a PyTree so that we can pass it from the forward to the backward
  # pass in a custom VJP.
  vjp_ =  Partial(partial(unbound_vjp, pvals, jaxpr), consts)
  if not has_aux:
    return out_primals, vjp_
  else:
    return out_primals, vjp_, aux


# NOTE: The FIXMEs below are caused by primal/tangent mixups (type
# errors if you will)
def backward_pass(jaxpr: core.Jaxpr, reduce_axes, transform_stack,
                  consts, primals_in, cotangents_in):
  if all(type(ct) is Zero for ct in cotangents_in) and not jaxpr.effects:
    return map(lambda v: Zero(v.aval), jaxpr.invars)

  def write_cotangent(prim, v, ct):
    # assert v not in primal_env
    assert ct is not Zero, (prim, v.aval)  # check for an old harmless type error
    if ct is None or type(v) is Literal:
      return
    if type(ct) is Zero:
      # FIXME: This triggers a lot of failures!
      # assert v.aval == ct.aval, (prim, v.aval, ct.aval)
      return
    axes_to_reduce = tuple(axis_name for axis_name in reduce_axes
                           if axis_name in core.get_aval(ct).named_shape
                           and axis_name not in v.aval.named_shape)
    if axes_to_reduce:
      ct = jax.lax.psum(ct, axis_name=axes_to_reduce)
    ct_env[v] = add_tangents(ct_env[v], ct) if v in ct_env else ct
    # TODO(mattjj): add back these checks for dynamic shapes
    # if config.jax_enable_checks:
    #   ct_aval = core.get_aval(ct_env[v])
    #   joined_aval = core.lattice_join(v.aval, ct_aval).strip_weak_type().strip_named_shape()
    #   assert v.aval.strip_weak_type().strip_named_shape() == joined_aval, (prim, v.aval, ct_aval)

  def read_cotangent(v):
    return ct_env.pop(v, Zero(v.aval))

  def read_primal(v):
    if type(v) is Literal:
      return v.val
    else:
      a = v.aval
      if type(a) is core.DShapedArray:
        shape = [primal_env[d] if type(d) is core.Var else d for d in a.shape]
        a = a.update(shape=tuple(shape))
      return primal_env.get(v, UndefinedPrimal(a))

  def write_primal(v, val):
    if not is_undefined_primal(val):
      primal_env[v] = val

  primal_env: Dict[Any, Any] = {}
  map(write_primal, jaxpr.constvars, consts)
  # FIXME: invars can contain both primal and tangent values, and this line
  #        forces primal_in to contain UndefinedPrimals for tangent values!
  map(write_primal, jaxpr.invars, primals_in)

  ct_env: Dict[Any, Any] = {}
  ctx = (source_info_util.transform_name_stack('transpose') if transform_stack
         else contextlib.nullcontext())
  with ctx:
    map(partial(write_cotangent, 'outvars'), jaxpr.outvars, cotangents_in)
    for eqn in jaxpr.eqns[::-1]:
      invals = map(read_primal, eqn.invars)
      if eqn.primitive.multiple_results:
        cts_in = map(read_cotangent, eqn.outvars)
      else:
        cts_in, = map(read_cotangent, eqn.outvars)
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
        if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
          cts_in_avals = [v.aval for v in eqn.outvars]
          params = dict(eqn.params)
          call_jaxpr = params.pop('call_jaxpr')
          cts_out = get_primitive_transpose(eqn.primitive)(
              params, call_jaxpr, invals, cts_in, cts_in_avals, reduce_axes)
        elif eqn.primitive in reducing_transposes:
          cts_out = reducing_transposes[eqn.primitive](
              reduce_axes, cts_in, *invals, **eqn.params)
        else:
          cts_out = get_primitive_transpose(eqn.primitive)(
              cts_in, *invals, **eqn.params)
        cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
        # FIXME: Some invars correspond to primals!
        map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out)

  cotangents_out = map(read_cotangent, jaxpr.invars)
  return cotangents_out


