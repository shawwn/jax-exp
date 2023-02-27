import numpy as np
import transformers
import functools
import abc

memoize = functools.lru_cache(maxsize=None)

class Codebook:
    def __init__(self, tokens):
        self.tokens = np.asarray(tokens, dtype=np.int64)

    @abc.abstractmethod
    def encode(self, text): ...

    @abc.abstractmethod
    def decode(self, tokens): ...

    @property
    def size(self):
        return len(self.tokens)

# import tokenizers.models
# import tokenizers.pre_tokenizers
# import tokenizers.decoders
#
# @memoize
# def gpt2_tokenizer(path=os.path.dirname(__file__)):
#     vocab_path = os.path.join(path, 'vocab.bpe')
#     encoder_path = os.path.join(path, 'encoder.json')
#     tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE.from_file(encoder_path, vocab_path))
#     tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(False)
#     tokenizer.decoder = tokenizers.decoders.ByteLevel()
#     return tokenizer

@memoize
def gpt2_tokenizer():
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    return tokenizer.backend_tokenizer

class CodebookGPT2(Codebook):
    def __init__(self):
        super().__init__(np.arange(50257))
        self.tokenizer = gpt2_tokenizer()

    def encode(self, text):
        tokens = self.tokenizer.encode(text).ids
        return np.array(tokens, dtype=np.uint16)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

def make_codebook_gpt2():
    return CodebookGPT2()

@memoize
def process_dataset(text_file):
    codebook = make_codebook_gpt2()
    if text_file.endswith('.tok16'):
        tokens = np.memmap(text_file, dtype=np.uint16)
    else:
        with open(text_file, encoding='utf-8') as f:
            text = f.read()
        tokens = codebook.encode(text)
    return tokens, codebook

def reshape_seqs(X, seq_size):
    n = (len(X) // seq_size) * seq_size
    return X[:n].reshape((-1, seq_size))

def train_test_split(codebook, text, n_ctx, train_percentage=0.75):
    chunksize = n_ctx + 1
    X_bt = reshape_seqs(text, chunksize)
    train_end = int(len(X_bt) * train_percentage)
    Xtr_bt = X_bt[:train_end]
    Xte_bt = X_bt[train_end:]
    return Xtr_bt, Xte_bt

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
