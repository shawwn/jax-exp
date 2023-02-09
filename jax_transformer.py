"""
Jax implementation of transformer language model loosely based on
https://github.com/openai/finetune-transformer-lm/
character-based model with simplifications
"""
# from jax.example_libraries import optimizers
import optimizers
import jax._src.nn.functions as F
import dataset_util
import functools
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import re
import jax
import jaxy
import time
import tqdm
from typing import *

# ================================================================
# Tf-like framework for Jax
# ================================================================

def create_root_context():
    return VariableContext({}, '')

class VariableContext(object):
    def __init__(self, name2val, prefix, allow_new=True):
        self.name2val = name2val
        self.prefix = prefix
        self.allow_new = allow_new

    def scope(self, name):
        return VariableContext(self.name2val, self._join(self.prefix, name), self.allow_new)

    def get_variable(self, name, initializer):
        return self.get_variable_absolute(
            name=self._join(self.prefix, name), 
            initializer=initializer)

    def get_variable_absolute(self, name, initializer):
        if name not in self.name2val:
            assert self.allow_new
            val = initializer()
            assert type(val) == np.ndarray and val.dtype == np.float32
            self.name2val[name] = val

        return self.name2val[name]

    def _join(self, *xs):
        return '/'.join(xs)

    def variables_list(self):
        return list(self.name2val.values())

    def replace_with_list(self, newlist):
        assert len(newlist) == len(self.name2val)
        name2val = {k : v for (k, v) in zip(self.name2val.keys(), newlist)}
        return VariableContext(name2val, self.prefix, self.allow_new)

    def transform(self, func):
        @functools.wraps(func)
        def transformed(params, *args, **kws):
            cx = self.replace_with_list(params)
            return func(cx, *args, **kws)
        return transformed


def print_variables(cx):
    for (name, val) in sorted(cx.name2val.items()):
        print(f'{name:20s} {str(val.shape):20s} {str(val.dtype):20s}')

# End framework 
# ----------------------------------------

def normax(shape, axis):
    out = npr.randn(*shape).astype(jnp.float32)
    out /= np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
    return out

def normc(*shape):
    return normax(shape, axis=0)

def randn(shape, stddev):
    return npr.randn(*shape).astype(jnp.float32) * stddev

def gelu(x):
    return 0.5*x*(1 + jnp.tanh(0.79788 * (x + 0.044715 * x ** 3)))

def _norm(x, *, axis, g=None, b=None, e=1e-5):
    u = jnp.mean(x, axis=axis, keepdims=True)
    s = jnp.mean(jnp.square(x - u), axis=axis, keepdims=True)
    x = (x - u) / jnp.sqrt(s + e)
    if g is not None and b is not None:
        x = x * g + b
    return x

def norm(cx, x, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda : np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda : np.zeros(n_state, 'f'))
    return _norm(x, g=g, b=b, axis=axis)

def mask_attn_weights(w):
    n = w.shape[-1]
    b = jnp.tril(jnp.ones((n, n)))
    # b = jnp.reshape(b, (1, 1, n, n))
    while len(b.shape) < len(w.shape):
        b = b[None, ...]
    w = w * b - 1e9 * (1 - b)
    return w

def _attn(Q_bhtr, K_bhrt, V_bhtr):
    R = Q_bhtr.shape[-1]
    W_bhtt = jnp.matmul(Q_bhtr, K_bhrt) / jnp.sqrt(R)
    W_bhtt = mask_attn_weights(W_bhtt)
    W_bhtt = F.softmax(W_bhtt, axis=-1)
    A_bhtr = jnp.matmul(W_bhtt, V_bhtr)
    return A_bhtr

def dense(cx, X_btk, F):
    *BT, K = X_btk.shape
    X_bt_k = jnp.reshape(X_btk, (-1, K))
    W_kf = cx.get_variable("w", initializer=lambda : normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda : np.zeros(F, 'f'))
    Y_bt_f = jnp.matmul(X_bt_k, W_kf) + b_f
    return jnp.reshape(Y_bt_f, (*BT, F))

def attn(cx, X_btk, n_state, n_head):
    *BT, _K = X_btk.shape
    assert n_state % n_head==0
    QKV_b_t_3s = dense(cx.scope('c_attn'), X_btk, n_state * 3)
    QKV_b_t_3h_r = jnp.reshape(QKV_b_t_3s, (*BT, 3 * n_head, n_state // n_head))
    # Q_bthr, K_bthr, V_bthr = jnp.split(QKV_b_t_3h_r, 3, axis=2)
    # Q_bhtr = jnp.transpose(Q_bthr, (0, 2, 1, 3))
    # V_bhtr = jnp.transpose(V_bthr, (0, 2, 1, 3))
    # K_bhrt = jnp.transpose(K_bthr, (0, 2, 3, 1))
    Q_bthr, K_bthr, V_bthr = jnp.split(QKV_b_t_3h_r, 3, axis=-2)
    Q_bhtr = jnp.swapaxes(Q_bthr, -3, -2)
    V_bhtr = jnp.swapaxes(V_bthr, -3, -2)
    K_bhtr = jnp.swapaxes(K_bthr, -3, -2)
    K_bhrt = jnp.swapaxes(K_bhtr, -2, -1)
    A_bhtr = _attn(Q_bhtr, K_bhrt, V_bhtr)
    # A_bthr = jnp.transpose(A_bhtr, (0, 2, 1, 3))
    A_bthr = jnp.swapaxes(A_bhtr, -3, -2)
    A_bts = jnp.reshape(A_bthr, (*BT, n_state))
    P_bts = dense(cx.scope('c_proj'), A_bts, n_state)
    return P_bts

def mlp(cx, X_bts, *, n_hid):
    S = X_bts.shape[-1]
    H_bth = F.gelu(dense(cx.scope('c_fc'), X_bts, n_hid))
    Y_bts = dense(cx.scope('c_proj'), H_bth, S)
    return Y_bts

def block(cx, X_bts, *, n_head):
    *_BT, S = X_bts.shape
    A_bts = attn(cx.scope('attn'), norm(cx.scope('ln_1'), X_bts), S, n_head)
    X_bts = X_bts + A_bts
    M_bts = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), X_bts), n_hid=S * 4)
    X_bts = X_bts + M_bts
    return X_bts

def transformer(cx, tok_bt, *, n_vocab, n_head, n_layer, n_ctx, n_embd):
    tok_bt = jnp.asarray(tok_bt)
    BT = tok_bt.shape
    pos_bt = jax.lax.broadcasted_iota(jnp.int32, BT, len(BT) - 1)
    tokenembs_qe = cx.get_variable('tokenembs', initializer=lambda: normc(n_vocab, n_embd) * 0.1)
    posembs_pe = cx.get_variable('posembs', initializer=lambda: normc(n_ctx, n_embd) * 0.1)
    tokenemb_bte = tokenembs_qe[tok_bt]
    posemb_bte = posembs_pe[pos_bt]
    last_bts = tokenemb_bte + posemb_bte
    for layer in range(n_layer):
        last_bts = block(cx.scope(f'h{layer:02d}'), last_bts, n_head=n_head)
    logits_btq = jnp.matmul(last_bts, tokenembs_qe.T)
    logprobs_btq = F.log_softmax(logits_btq)
    return logprobs_btq

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('text_files', nargs='+')
    parser.add_argument('--load_model', default=False)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-d', '--desc', '--description', type=str, default="")
    parser.add_argument('--n_vocab', type=int, default=dataset_util.make_codebook_gpt2().size)
    parser.add_argument('--n_ctx', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--fps', type=int, default=5, help="The number of times per second that the status bar shows loss values, etc")
    parser.add_argument('--adam_eps', type=float, default=1e-8, help="Adam's epsilon parameter")
    parser.add_argument('--adam_b1', type=float, default=0.9, help="Adam's beta1 parameter")
    parser.add_argument('--adam_b2', type=float, default=0.999, help="Adam's beta2 parameter")
    parser.add_argument('--dtype', type=str, default='fp32', help="dtype of params. fp32, fp16, or bf16")
    args = parser.parse_args()
    if args.seed < 0:
        args.seed = npr.randint(2**31)
    npr.seed(args.seed)
    text, codebook, Xtr_bt, Xte_bt = dataset_util.load_dataset(text_file := np.random.choice(args.text_files), args.n_ctx)

    model = functools.partial(
        transformer,
        n_vocab=codebook.size,
        n_ctx=args.n_ctx,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    root_cx = create_root_context()
    def train_example_count():
        return Xtr_bt.shape[0]

    def train_token_count():
        return np.prod(Xtr_bt.shape)

    def loss_batch(cx, XY_bt):
        X_bt, Y_bt = XY_bt[:, :-1], XY_bt[:, 1:]
        logprobs_btq = model(cx, X_bt)
        loglosses_btq = -logprobs_btq
        return loglosses_btq

    def loss_std(cx, XY_bt):
        X_bt, Y_bt = XY_bt[..., :-1], XY_bt[..., 1:]
        BT = np.prod(X_bt.shape)
        logprobs_btq = model(cx, X_bt)
        loglosses_bt = - logprobs_btq.reshape((BT, -1))[jnp.arange(BT), Y_bt.reshape((-1,))]
        return loglosses_bt.mean()

    def loss(cx, XY_bt):
        X_bt, Y_bt = XY_bt[..., :-1], XY_bt[..., 1:]
        BT = np.prod(X_bt.shape)
        logprobs_btq = model(cx, X_bt)
        loglosses_btq = -logprobs_btq
        Y_bt_ = Y_bt.reshape((-1,))
        loglosses_bt_q = loglosses_btq.reshape((BT, -1))
        loglosses_bt_ = loglosses_bt_q[jnp.arange(BT), Y_bt_]
        return loglosses_bt_.mean()

    def loss2(params, XY_bt):
        cx = root_cx.replace_with_list(params)
        return loss(cx, XY_bt)


    loss(root_cx, Xtr_bt[:args.batch_size]) # Just create variables
    root_cx.allow_new = False
    init_params = root_cx.variables_list()
    if args.dtype == 'fp32':
        pass
    elif args.dtype == 'fp16':
        init_params = jax.tree_map(lambda x: x.astype(jnp.float16), init_params)
    elif args.dtype == 'bf16':
        init_params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), init_params)
    else:
        assert False, "Invalid --dtype"

    root_cx = root_cx.replace_with_list(init_params)
    print_variables(root_cx)
    def print_hparams():
        print('=' * 50)
        for k, v in args.__dict__.items():
            print(f'{k}: {v!r}')
        print('=' * 50)
    print_hparams()

    # opt_init, opt_update, get_params = optimizers.adam_std(step_size=args.lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps)
    # opt_init, opt_update, get_params = optimizers.adamsp(step_size=args.lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps)
    # opt_init, opt_update, get_params = optimizers.adamsp_v2(step_size=args.lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps)
    # opt_init, opt_update, get_params = optimizers.adamsp_volavg(step_size=args.lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps)
    opt_init, opt_update, get_params = optimizers.adamsp_1bit(step_size=args.lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps)
    opt_state = opt_init(init_params)

    def each(f, x):
        state = ()
        state, y = jax.lax.scan(lambda state, x: (state, f(x)), state, x)
        return y

    def grad(loss, params):
        return functools.partial(jax.grad(root_cx.transform(loss)), params)

    def valgrad(loss, params):
        return functools.partial(jax.value_and_grad(root_cx.transform(loss)), params)



    # @jax.jit
    def update_old(i, opt_state, batch):
        XY, = batch
        params = get_params(opt_state)
        # fn = jax.value_and_grad(root_cx.transform(loss))
        # v, g = jaxy.value_and_grad(root_cx.transform(loss))(params, XY)
        # return 1, [v], opt_update(i, g, opt_state)
        # vs, gs = jax.lax.scan(lambda carry, x: (carry, jax.value_and_grad(root_cx.transform(loss))(params, x)), (), XY[:, None, :])
        # vs, gs = each(valgrad(loss, params), XY[:, None, :])
        vs, gs = each(valgrad(loss, params), XY)
        n, dn = XY.shape[0], 1
        # n, dn = 1, 0
        # new_state, _ = jax.lax.scan(lambda opt_state, g: (opt_update(i, g, opt_state), None), opt_state, gs)
        def inner(aux, g):
            i, opt_state = aux
            g = jax.tree_map(lambda grad: grad / n, g)
            new_state = opt_update(i, g, opt_state)
            return (i+dn, new_state), None
        (i, new_state), _ = jax.lax.scan(inner, (i, opt_state), gs)
        return n, vs, new_state

    # @jax.jit
    def update_old(i, opt_state, batch):
        XY, = batch
        params = get_params(opt_state)
        lossfn = root_cx.transform(loss)
        v, g = jax.value_and_grad(lossfn)(params, XY)
        # n, dn = 1, 0
        # n, dn = XY.shape[0], 1
        # n, dn = 200**2, 1
        # g = jax.tree_map(lambda grad: grad * n, g)
        new_state = opt_update(i, g, opt_state)
        return 1, [v], new_state

    @jax.jit
    def update(i, opt_state, batch):
        XY, = batch
        params = get_params(opt_state)
        v, g = jax.value_and_grad(loss2)(params, XY)
        return v, opt_update(i, g, opt_state)

    bars = {}

    def make_pbar(*argv, **kws):
        make = tqdm.trange if argv else tqdm.tqdm
        interval = 1 / args.fps
        pos = kws.pop('position', len(bars))
        bar = make(*argv,
                   position=pos,
                   dynamic_ncols=True,
                   # ncols=0,
                   mininterval=interval,
                   maxinterval=interval,
                   leave=True,
                   smoothing=0.8,
                   **kws)
        assert pos not in bars, f"A progress bar is already at position {pos}"
        bars[pos] = bar
        return bar

    def timestamp():
        now = time.time()
        n = int(now - start)
        h = n // (60*60)
        m = (n // 60) % 60
        s = n % 60
        def num(x, suffix):
            if x != 0:
                return f'{x: >2}{suffix}'
            else:
                return ' ' * (len(suffix) + 2)
        ts = num(h, "h") + num(m, "m") + num(s, "s")
        ts = f'[{h:02}:{m:02}:{s:02}] '
        return f'{ts}  time: {now - start: < 8,.2f}'

    def stats(loss):
        return dict(steps=f'{pstep.n:<8,}',
                    examples=f'{pexamples.n:<10,}',
                    loss=f'{loss:<8.4f}',
                    wisdom=f'{10/loss:<8,.2f}',
                    )
    def hparams():
        return dict(
            batch=f'{args.batch_size}',
            seed=f'{args.seed}',
            lr=f'{args.lr:.1e}',
            b1=args.adam_b1,
            b2=args.adam_b2,
            eps=args.adam_eps,
            n_vocab=codebook.size,
            n_ctx=args.n_ctx,
            n_head=args.n_head,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
    def tagline(h: dict):
        return ' '.join([f'{k}: {v}' for k, v in h.items()])

    start = time.time()
    loss_sum = 0.0
    loss_n = 0
    show_n = int(start)
    pstep = make_pbar(unit='steps', unit_scale=True)
    pexamples = make_pbar(unit='examples', unit_scale=True)
    ptokens = make_pbar(unit='tokens', unit_scale=True)
    pepoch = make_pbar(unit='epochs')
    pwrite = pstep
    while True:
        ptokens.reset(train_token_count())
        pepoch.set_postfix(dict(desc=repr(args.desc), text_file=text_file))
        for XY in dataset_util.iterbatches(Xtr_bt, batch_size=args.batch_size, include_final_partial_batch=False):
            try:
                lossvals, opt_state = update(pstep.n, opt_state, XY)
                k, lossvals = 1, [lossvals]
                # k, lossvals, opt_state = update(pstep.n, opt_state, XY)
                # k = np.asarray(k).item()
                loss_sum += sum(lossvals)
                loss_n += len(lossvals)
                pstep.set_postfix(stats(max(lossvals)), refresh=False)
                pstep.update(k)
                pexamples.set_postfix(hparams(), refresh=False)
                pexamples.update(args.batch_size)
                ptokens.update(args.batch_size * args.n_ctx)
                def need_show(n, k):
                    if n <= 1_000 and (n//k) % (50//((k//4)+1)) == 0:
                        return True
                    if int(time.time()) > show_n and (n//k) % (250//((k//4)+1)) == 0:
                        return True
                if need_show(pstep.n, k):
                    pwrite.write(timestamp() + " " + tagline(stats(loss_sum / loss_n)))
                    loss_sum = 0.0
                    loss_n = 0
                    show_n = int(time.time())
            except KeyboardInterrupt:
                with pwrite.external_write_mode():
                    print("")
                    print(repr(args.desc), hparams())
                    breakpoint()
        pepoch.update(1)
        # new epoch; load a different text file.
        text_file2 = np.random.choice(args.text_files)
        if text_file2 != text_file:
            text_file = text_file2
            text, codebook, Xtr_bt, Xte_bt = dataset_util.load_dataset(text_file, args.n_ctx)

if __name__ == '__main__':
    main()