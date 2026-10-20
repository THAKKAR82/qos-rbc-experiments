"""Frozen training/validation access only; causal document-local features."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.text_preprocessing import make_tfidf_vectorizer
from .state import inventory, totals

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / 'validation/cfpb/20261014T192947Z-7b6668f9'
TASKS = {
    'credit_vs_checking': dict(positive=1, negative=2, N=7265, validation_N=1588,
                              audit_id='credit_card__vs__checking_or_savings_account'),
    'mortgage_vs_rest': dict(positive=3, negative=None, N=13459, validation_N=3033,
                            audit_id='mortgage__vs__rest'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def task_mask(labels, task, tasks=None):
    t = (TASKS if tasks is None else tasks)[task]
    return np.ones(len(labels), bool) if t['negative'] is None else np.isin(labels, [t['positive'], t['negative']])


class Encoder:
    __slots__ = ('representation', 'vectorizer', 'stop_words', 'pattern')

    def __init__(self, representation, preparation=PREPARATION):
        if representation not in ('A_frozen', 'B_causal'):
            raise ValueError('Unknown representation')
        self.representation = representation
        self.stop_words = ENGLISH_STOP_WORDS
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        self.vectorizer = None
        if representation == 'A_frozen':
            with (preparation / 'vocabulary.csv').open() as f:
                vocabulary = {r['token']: i for i, r in enumerate(csv.DictReader(f))}
            self.vectorizer = make_tfidf_vectorizer().set_params(vocabulary=vocabulary)
            self.vectorizer.idf_ = np.load(preparation / 'idf.npy', allow_pickle=False)
            self.vectorizer.transform([''])  # Finalize fixed-vocabulary bookkeeping, no fitting.

    def encode(self, text):
        if self.vectorizer is not None:
            row = self.vectorizer.transform([text]).tocsr()
            row.sum_duplicates(); row.eliminate_zeros(); row.sort_indices()
            keys = np.zeros((len(row.data), 2), dtype=np.uint64)
            keys[:, 0] = row.indices
            return keys, row.data.astype(np.float64)
        words = [w for w in self.pattern.findall(text.lower()) if w not in self.stop_words]
        counts = Counter(words)
        counts.update(a + ' ' + b for a, b in zip(words, words[1:]))
        # No cross-document vocabulary, document frequencies, label counters,
        # adaptive token admission, or observed-feature cache.
        keyed = {}
        for term, value in counts.items():
            digest = hashlib.blake2b(term.encode('utf-8'), digest_size=16, person=b'cfpb-stream-v1').digest()
            key = (int.from_bytes(digest[:8], 'little'), int.from_bytes(digest[8:], 'little'))
            if key in keyed:
                raise ValueError('Distinct within-document terms collided in 128-bit identity')
            keyed[key] = value
        pairs = sorted(keyed)
        keys = np.array(pairs, dtype=np.uint64).reshape(-1, 2)
        values = np.array([keyed[k] for k in pairs], dtype=np.float64)
        norm = np.linalg.norm(values)
        if norm:
            values /= norm
        return keys, values

    def state_inventory(self):
        return inventory([('preprocessing', 'encoder', self)], compact=True)


def write_spool(path, rows):
    """Offline access infrastructure; caller reports data/offset disk bytes."""
    path = Path(path)
    offsets = []
    with path.open('wb') as f:
        for row in rows:
            offsets.append(f.tell())
            f.write((json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n').encode())
    np.save(path.with_suffix('.offsets.npy'), np.asarray(offsets, np.int64), allow_pickle=False)
    return len(offsets)


class Stream:
    """One forward learner pass. Random replay explicitly owns N-sized access arrays."""
    __slots__ = ('file', 'path', 'n', 'cursor', 'order', 'offsets', 'mode', 'role', 'last_key', 'seed')

    def __init__(self, path, n, *, role, order='chronological', seed=42, cursor=0):
        if role not in ('train', 'validation'):
            raise ValueError('Test access is prohibited in this pilot')
        if order not in ('chronological', 'random') or not 0 <= cursor <= n:
            raise ValueError('Invalid stream contract')
        self.path, self.n, self.cursor = str(path), int(n), int(cursor)
        self.role, self.mode, self.seed, self.last_key = role, order, int(seed), None
        self.file = open(path, 'rb', buffering=8192)
        self.order = self.offsets = None
        if order == 'random':
            self.offsets = np.load(Path(path).with_suffix('.offsets.npy'), allow_pickle=False)
            if self.offsets.shape != (n,):
                raise ValueError('Offset length mismatch')
            rng = np.random.default_rng(np.random.SeedSequence([seed, 619]))
            self.order = rng.permutation(n).astype(np.int32)
        elif cursor:
            # Restore seek position without replaying observations to a learner.
            # Read a single int64 from the offset file, never load the whole array.
            with Path(path).with_suffix('.offsets.npy').open('rb') as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError('Unsupported restart-index version')
                if shape != (n,) or dtype != np.dtype('int64') or fortran:
                    raise ValueError('Invalid restart index')
                if cursor < n:
                    f.seek(cursor * 8, 1)
                    self.file.seek(int(np.frombuffer(f.read(8), dtype=np.int64)[0]))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= self.n:
            raise StopIteration
        if self.order is not None:
            self.file.seek(int(self.offsets[self.order[self.cursor]]))
        line = self.file.readline()
        if not line:
            raise ValueError('Premature end of stream')
        row = json.loads(line)
        lower, upper = ('2026-01-01', '2026-01-31') if self.role == 'train' else ('2026-02-01', '2026-02-14')
        if row['split'] != self.role or not lower <= row['date'] <= upper:
            raise ValueError('Future/incorrect-partition observation blocked')
        key = (row['date'], int(row['id']))
        if self.mode == 'chronological' and self.last_key is not None and key <= self.last_key:
            raise ValueError('Chronological access violation')
        self.last_key = key
        self.cursor += 1
        return row

    def close(self):
        self.file.close()

    def state_inventory(self):
        entries = inventory([('stream_access', 'stream', self),
                             ('stream_access', 'raw_file', self.file.raw)])
        # BufferedReader.__sizeof__ includes its 8 KiB buffer; the raw handle
        # is separate. OS filesystem/page cache belongs to system memory/RSS.
        return entries

    def state_bytes(self):
        return totals(self.state_inventory())['bytes']
