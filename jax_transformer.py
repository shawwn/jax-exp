"""
Jax implementation of transformer language model loosely based on
https://github.com/openai/finetune-transformer-lm/
character-based model with simplifications
"""
from __future__ import annotations
from jax.example_libraries import optimizers
import jax._src.nn.functions as F
import dataset_util
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import jax
import time
import tqdm
import argparse
from typing import cast

class Config(argparse.Namespace):
    seed: int
    lr: float
    adam_eps: float
    adam_b1: float
    adam_b2: float
    batch_size: int
    desc: str
    n_vocab: int
    n_ctx: int
    n_head: int
    n_layer: int
    n_embd: int
    fps: int
    rotary: bool

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('text_files', nargs='+')
        parser.add_argument('--load_model', default=False)
        parser.add_argument('--seed', type=int, default=-1)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--adam_eps', type=float, default=1e-8, help="Adam's epsilon parameter")
        parser.add_argument('--adam_b1', type=float, default=0.9, help="Adam's beta1 parameter")
        parser.add_argument('--adam_b2', type=float, default=0.999, help="Adam's beta2 parameter")
        parser.add_argument('-b', '--batch_size', type=int, default=64)
        parser.add_argument('-d', '--desc', '--description', type=str, default="")
        parser.add_argument('--n_vocab', type=int, default=dataset_util.make_codebook_gpt2().size)
        parser.add_argument('--n_ctx', type=int, default=64)
        parser.add_argument('--n_head', type=int, default=4)
        parser.add_argument('--n_layer', type=int, default=4)
        parser.add_argument('--n_embd', type=int, default=128)
        parser.add_argument('--fps', type=int, default=5, help="The number of times per second that the status bar shows loss values, etc")
        parser.add_argument('--rotary', action='store_true', help="Whether to use rotary embeddings")
        return cast(cls, parser.parse_args())

# ================================================================
# Tf-like framework for Jax
# ================================================================

def create_root_context(cfg: Config, prefix: str):
    return VariableContext(cfg, {}, prefix, allow_new=True)

@jax.tree_util.register_pytree_node_class
class VariableContext:
    def __init__(self, cfg: Config, name2val, prefix, allow_new):
        self.cfg = cfg
        self.name2val = name2val
        self.prefix = prefix
        self.allow_new = allow_new

    def scope(self, name):
        return VariableContext(self.cfg, self.name2val, self._join(self.prefix, name), self.allow_new)

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
        return jnp.asarray(self.name2val[name])

    def _join(self, *xs):
        return '/'.join(xs)

    def variables_list(self):
        return list(self.name2val.values())

    def replace_with_list(self, newlist):
        assert len(newlist) == len(self.name2val)
        name2val = {k : v for (k, v) in zip(self.name2val.keys(), newlist)}
        return VariableContext(self.cfg, name2val, self.prefix, self.allow_new)

    def tree_flatten(self):
        return self.variables_list(), self

    @classmethod
    def tree_unflatten(cls, parent: VariableContext, values):
        if not parent.allow_new:
            return parent.replace_with_list(values)
        return parent

    def __repr__(self):
        return (f'VariableContext(len(name2val)={len(self.name2val)}, '
                f'prefix={self.prefix!r}, allow_new={self.allow_new})')

def print_variables(cx: VariableContext):
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

def _norm(x, *, axis, g=None, b=None, e=1e-5):
    u = jnp.mean(x, axis=axis, keepdims=True)
    s = jnp.mean(jnp.square(x - u), axis=axis, keepdims=True)
    x = (x - u) / jnp.sqrt(s + e)
    if g is not None and b is not None:
        x = x * g + b
    return x

def norm(cx: VariableContext, x, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda: np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda: np.zeros(n_state, 'f'))
    return _norm(x, g=g, b=b, axis=axis)

def mask_attn_weights(w):
    n = w.shape[-1]
    b = jnp.tril(jnp.ones_like(w, shape=(n, n), dtype=w.dtype))
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

def dense(cx: VariableContext, X_btk, F):
    *BT, K = X_btk.shape
    X_bt_k = jnp.reshape(X_btk, (-1, K))
    W_kf = cx.get_variable("w", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: np.zeros(F, 'f'))
    Y_bt_f = jnp.matmul(X_bt_k, W_kf) + b_f
    return jnp.reshape(Y_bt_f, (*BT, F))

def attn_qkv(cx: VariableContext, Q_bthd, K_bthd, V_bthd):
    Q_bhtd = jnp.swapaxes(Q_bthd, -3, -2)
    K_bhtd = jnp.swapaxes(K_bthd, -3, -2)
    K_bhdt = jnp.swapaxes(K_bhtd, -2, -1)
    V_bhtd = jnp.swapaxes(V_bthd, -3, -2)
    R = Q_bhtd.shape[-1]
    invsqrt_keysize = jnp.array(1.0 / np.sqrt(R), dtype=Q_bhtd.dtype)
    W_bhtt = jnp.matmul(Q_bhtd, K_bhdt) * invsqrt_keysize
    W_bhtt = mask_attn_weights(W_bhtt)
    W_bhtt = F.softmax(W_bhtt, axis=-1)
    A_bhtd = jnp.matmul(W_bhtt, V_bhtd)
    A_bthd = jnp.swapaxes(A_bhtd, -3, -2)
    return A_bthd

def attn(cx: VariableContext, X_bts, n_state, n_head):
    assert n_state % n_head == 0
    *BT, _S = X_bts.shape
    QKV_b_t_3s = dense(cx.scope('c_attn'), X_bts, n_state * 3)
    QKV_b_t_3h_d = jnp.reshape(QKV_b_t_3s, (*BT, 3 * n_head, n_state // n_head))
    Q_bthd, K_bthd, V_bthd = jnp.split(QKV_b_t_3h_d, 3, axis=-2)
    A_bthd = attn_qkv(cx, Q_bthd, K_bthd, V_bthd)
    A_bts = jnp.reshape(A_bthd, (*BT, n_state))
    P_bts = dense(cx.scope('c_proj'), A_bts, n_state)
    return P_bts

def mlp(cx: VariableContext, X_bts, *, n_hid):
    S = X_bts.shape[-1]
    H_bth = dense(cx.scope('c_fc'), X_bts, n_hid)
    H_bth = F.gelu(H_bth)
    X_bts = dense(cx.scope('c_proj'), H_bth, S)
    return X_bts

def block(cx: VariableContext, X_bts):
    *_BT, S = X_bts.shape
    X_bts = X_bts + attn(cx.scope('attn'), norm(cx.scope('ln_1'), X_bts), S, cx.cfg.n_head)
    X_bts = X_bts + mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), X_bts), n_hid=S * 4)
    return X_bts

def transformer(cx: VariableContext, tok_bt):
    tok_bt = jnp.asarray(tok_bt)

    # Vocab embedding
    tokenembs_qe = cx.get_variable('wte', initializer=lambda: normc(cx.cfg.n_vocab, cx.cfg.n_embd) * 0.1)
    tokenemb_bte = tokenembs_qe[tok_bt]
    H_bts = tokenemb_bte

    # Position embedding
    if not cx.cfg.rotary:
        BT = tok_bt.shape
        pos_bt = jax.lax.broadcasted_iota(jnp.int32, BT, len(BT) - 1)
        posembs_pe = cx.get_variable('wpe', initializer=lambda: normc(cx.cfg.n_ctx, cx.cfg.n_embd) * 0.1)
        posemb_bte = posembs_pe[pos_bt]
        H_bts = H_bts + posemb_bte
    else:
        raise NotImplementedError("TODO: Rotary embeddings")

    # Pass the embedding through each block
    block_fn = jax.checkpoint(block)
    for layer in range(cx.cfg.n_layer):
        H_bts = block_fn(cx.scope(f'h{layer}'), H_bts)

    # Do a final normalization
    H_bts = norm(cx.scope('ln_f'), H_bts)

    # Multiply the activation by the transpose of the token embedding to get logits
    logits_btq = jnp.matmul(H_bts, tokenembs_qe.T)

    # Return the logits
    return logits_btq

def sparse_softmax_cross_entropy_with_logits(logits_btq, labels_bt):
    BT = np.prod(np.shape(labels_bt))
    logprobs_btq = F.log_softmax(logits_btq)
    loglosses_btq = -logprobs_btq
    labels_bt_ = labels_bt.reshape((-1,))
    loglosses_bt_q = loglosses_btq.reshape((BT, -1))
    loglosses_bt_ = loglosses_bt_q[jnp.arange(BT), labels_bt_]
    return loglosses_bt_

def loss(cx: VariableContext, XY_bt):
    X_bt = XY_bt[:, :-1]
    Y_bt = XY_bt[:, 1:]
    logits_btq = transformer(cx, X_bt)
    loglosses_bt_ = sparse_softmax_cross_entropy_with_logits(logits_btq, Y_bt)
    return loglosses_bt_.mean()

def main():
    cfg = Config.parse_args()
    if cfg.seed < 0:
        cfg.seed = npr.randint(2**31)
    npr.seed(cfg.seed)
    text, codebook, Xtr_bt, Xte_bt = dataset_util.load_dataset(text_file := np.random.choice(cfg.text_files), cfg.n_ctx)
    cfg.n_vocab = codebook.size
    root_cx = create_root_context(cfg, 'model')

    def train_example_count():
        return Xtr_bt.shape[0]

    def train_token_count():
        return np.prod(Xtr_bt.shape)

    jax.jit(loss).lower(root_cx, Xtr_bt[:cfg.batch_size]) # Just create variables
    root_cx.allow_new = False
    print_variables(root_cx)
    init_params = root_cx.variables_list()
    def print_hparams():
        print('=' * 50)
        for k, v in cfg.__dict__.items():
            print(f'{k}: {v!r}')
        print('=' * 50)
    print_hparams()

    opt_init, opt_update, get_params = optimizers.adam(step_size=cfg.lr, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps)
    opt_state = opt_init(init_params)

    @jax.jit
    def update(i, opt_state, batch):
        XY, = batch
        params = get_params(opt_state)
        cx = root_cx.replace_with_list(params)
        loss_v, cx_grad = jax.value_and_grad(loss)(cx, XY)
        g = cx_grad.variables_list()
        return loss_v, opt_update(i, g, opt_state)

    bars = {}

    def make_pbar(*argv, **kws):
        make = tqdm.trange if argv else tqdm.tqdm
        interval = 1 / cfg.fps
        pos = kws.pop('position', len(bars))
        bar = make(*argv,
                   position=pos,
                   dynamic_ncols=True,
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
            batch=f'{cfg.batch_size}',
            # seed=f'{cfg.seed}',
            lr=f'{cfg.lr:.1e}',
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            eps=cfg.adam_eps,
            n_vocab=codebook.size,
            n_ctx=cfg.n_ctx,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
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
        pepoch.set_postfix(dict(desc=repr(cfg.desc), seed=f'{cfg.seed}', text_file=text_file))
        for XY in dataset_util.iterbatches(Xtr_bt, batch_size=cfg.batch_size, include_final_partial_batch=False):
            try:
                lossval, opt_state = update(pstep.n, opt_state, XY)
                loss_sum += lossval
                loss_n += 1
                pstep.set_postfix(stats(lossval), refresh=False)
                pexamples.set_postfix(hparams(), refresh=False)
                pexamples.update(cfg.batch_size)
                ptokens.update(cfg.batch_size * cfg.n_ctx)
                pstep.update(1)
                def need_show(n):
                    if n <= 1_000 and n % 50 == 0:
                        return True
                    if int(time.time()) > show_n and n % 250 == 0:
                        return True
                if need_show(pstep.n):
                    pwrite.write(timestamp() + " " + tagline(stats(loss_sum / loss_n)))
                    loss_sum = 0.0
                    loss_n = 0
                    show_n = int(time.time())
            except KeyboardInterrupt:
                with pwrite.external_write_mode():
                    print("")
                    print(repr(cfg.desc), f'seed={cfg.seed}', hparams())
                    breakpoint()
        pepoch.update(1)
        # new epoch; load a different text file.
        text_file2 = np.random.choice(cfg.text_files)
        if text_file2 != text_file:
            text_file = text_file2
            text, codebook, Xtr_bt, Xte_bt = dataset_util.load_dataset(text_file, cfg.n_ctx)

if __name__ == '__main__':
    main()