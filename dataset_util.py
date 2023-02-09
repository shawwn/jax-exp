# import joblib
from tabulate import tabulate
import numpy as np
import io
import collections
import sys
import re
import functools

class Codebook(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def token2idx(self, token):
        return self.tokens.index(token)

    def idx2token(self, idx):
        return self.tokens[idx]

    def encode(self, text):
        return [self.token2idx(token) for token in text]

    @property
    def size(self):
        return len(self.tokens)


class CodebookGPT2(Codebook):
    def __init__(self):
        super().__init__(list(range(50257)))

# mem = joblib.Memory('/tmp/dataset_util')


# @mem.cache
@functools.lru_cache(maxsize=None)
def make_codebook(text):
    all_chars = list(sorted(set(text)))
    codebook = Codebook(all_chars)
    return codebook

def make_codebook_gpt2():
    return CodebookGPT2()

# @mem.cache
def get_zip_ratio(text):
    import zlib
    if isinstance(text, str):
        text = text.encode()
    smalltext = zlib.compress(text, level=-1)
    ratio = len(smalltext) / len(text)
    return ratio


def process_dataset(text_file, print_stats=False):
    if text_file.endswith('.tok16'):
        tokens = np.fromfile(text_file, dtype=np.uint16)
        return tokens, make_codebook_gpt2()
    if text_file.endswith('.tok32'):
        tokens = np.fromfile(text_file, dtype=np.uint32)
        return tokens, make_codebook_gpt2()
    try:
        with io.open(text_file, encoding='utf-8') as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        print('Unicode error; training as non-unicode', file=sys.stderr)
        with io.open(text_file, mode='rb') as f:
            text = f.read().strip()
    codebook = make_codebook(text)
    if print_stats:
        token2count = collections.Counter(text)
        counts = np.array([token2count[c] for c in codebook.tokens])
        probs = counts / counts.sum()
        print(tabulate(zip(map(repr, codebook.tokens), probs, map(int, counts)),
                       headers=['tokens', 'probs', 'counts'], floatfmt='.3e'))
        zipratio = get_zip_ratio(text)
        print(tabulate([
            ('Marg ent', (probs * np.log(1 / probs)).sum()),
            ('Zip', zipratio * np.log(256))
        ]))
    return text, codebook


def reshape_seqs(X, seq_size):
    n = (len(X) // seq_size) * seq_size
    return X[:n].reshape((-1, seq_size))


def train_test_split(codebook, text, n_ctx, train_percentage=0.75):
    if hasattr(text, 'shape'):
        X_bt = reshape_seqs(text, n_ctx)
        train_end = int(len(X_bt) * train_percentage)
        Xtr_bt = X_bt[:train_end]
        Xte_bt = X_bt[train_end:]
        return Xtr_bt, Xte_bt
    # TODO start at every character
    flatdata = np.array([codebook.token2idx(token) for token in text])
    splits = [mo.end() for mo in re.finditer(r'\n\n|\. |; |: ',text)]
    starts = np.concatenate([[0], splits])
    teststart = starts[int(len(starts) * train_percentage)]
    chunksize = n_ctx + 1
    starts_train = starts[starts + chunksize <= teststart]
    starts_test = starts[starts + chunksize <= len(flatdata)]
    return (np.array([flatdata[s: s + chunksize] for s in starts_train]),
            np.array([flatdata[s: s + chunksize] for s in starts_test]))

def load_dataset(text_file, n_ctx):
    text, codebook = process_dataset(text_file)
    Xtr_bt, Xte_bt = train_test_split(codebook, text, n_ctx)
    return text, codebook, Xtr_bt, Xte_bt


def iterbatches(*arrays, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
