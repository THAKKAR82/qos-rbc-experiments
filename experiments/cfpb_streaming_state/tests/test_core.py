import json
from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state.methods import Learner, IndexedHeap, budgeted
from experiments.cfpb_streaming_state.data import Encoder, Stream, write_spool
from experiments.cfpb_streaming_state.state import deep_bytes, inventory, totals


def keys(ids):
    return np.column_stack((np.asarray(ids, np.uint64), np.zeros(len(ids), np.uint64)))


def test_legacy_projection_rowwise_compatibility_and_charged_mapping():
    from scipy import sparse
    from src.sketches import sparse_jl_features
    from experiments.cfpb_streaming_state.compatibility import LegacyJL
    rng = np.random.default_rng(42)
    X = sparse.csr_matrix(rng.normal(size=(9, 23)).astype(np.float32))
    X.data[::3] = 0
    adapter = LegacyJL(23, 7)
    expected = sparse_jl_features(X, 7, seed=42)
    actual = sparse.vstack([adapter.row(X[i:i+1]) for i in range(len(X.indptr)-1)])
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    P = adapter.projection
    assert adapter.state_bytes() >= P.data.nbytes + P.indices.nbytes + P.indptr.nbytes


@pytest.mark.parametrize('method,depth', [('hash_sgd', 1), ('sparse_jl_sgd', 3)])
def test_projected_updates_against_dense_gradient_and_sklearn(method, depth):
    m = Learner(method, 7, depth=depth, lam=0., eta=.1)
    dense = np.zeros(len(m.weights))
    sk = SGDRegressor(loss='squared_error', penalty=None, fit_intercept=False,
                      learning_rate='constant', eta0=.1, shuffle=False, average=False, random_state=42)
    rng = np.random.default_rng(42)
    for _ in range(20):
        k = keys(np.arange(11)); x = rng.normal(size=11); x /= np.linalg.norm(x)
        y = rng.choice([-1, 1]); ix, signs = m.projection.hashes(k)
        R = np.zeros((len(dense), len(x)))
        for j in range(len(x)):
            R[ix[:, j], j] = signs[:, j] / np.sqrt(depth)
        z = R @ x
        pred = dense @ z
        dense -= .1 * (pred - y) * z
        sk.partial_fit(z[None, :], [y])
        assert m.update(k, x, y) == pytest.approx(pred, abs=1e-14)
        np.testing.assert_allclose(m.weights, dense, atol=1e-14, rtol=0)
        np.testing.assert_allclose(m.weights, sk.coef_, atol=1e-14, rtol=0)


def test_regularization_and_empty_row_against_dense_objective():
    m = Learner('hash_sgd', 5, eta=.2, lam=.03)
    ref = np.zeros(5)
    for t in range(30):
        k = keys([2, 7]) if t % 4 else keys([])
        x = np.array([.6, .8]) if len(k) else np.array([])
        ix, signs = m.projection.hashes(k)
        z = np.zeros(5)
        for i, s, v in zip(ix[0], signs[0], x):
            z[i] += s * v
        eta = .2 / (1 + .2 * .03 * t)
        ref -= eta * ((ref @ z - 1) * z + .03 * ref)
        m.update(k, x, 1)
        np.testing.assert_allclose(m.weights * m.scale, ref, atol=1e-14, rtol=0)


def test_fixed_heap_against_dictionary_top_candidates():
    heap = IndexedHeap(7); ref = {}; rng = np.random.default_rng(42)
    allocated = sum(a.nbytes for a in (heap.keys, heap.values, heap.heap, heap.positions, heap.table))
    for t in range(3000):
        key = (int(rng.integers(0, 40)), int(rng.integers(0, 3)))
        val = float(rng.choice([-3., -1., 0., 1., 3.]))
        if key in ref or len(ref) < 7:
            ref[key] = val
        else:
            smallest = min(ref, key=lambda k: (abs(ref[k]), k))
            if abs(val) > abs(ref[smallest]):
                del ref[smallest]; ref[key] = val
        heap.offer(*key, val)
        if t % 17 == 0:
            heap.check()
            actual = {tuple(map(int, heap.keys[i])): heap.values[i] for i in range(heap.n)}
            assert actual == ref
    assert sum(a.nbytes for a in (heap.keys, heap.values, heap.heap, heap.positions, heap.table)) == allocated


@pytest.mark.parametrize('method', ['awm', 'mission'])
def test_adaptive_updates_against_independent_dense_sketch_reference(method):
    m = Learner(method, 4, capacity=3, lam=.02 if method == 'awm' else 0.)
    S = np.zeros((3, 4)); active = {}; rng = np.random.default_rng(42)
    k = keys(range(9)); indices, signs = m.projection.hashes(k)
    buckets = indices % 4

    def query(j):
        return float(np.median([signs[r, j] * S[r, buckets[r, j]] for r in range(3)]))

    def add(j, value):
        for r in range(3):
            S[r, buckets[r, j]] += signs[r, j] * value

    def admit(j, value):
        if j in active or len(active) < 3:
            active[j] = value
            return True, None
        smallest = min(active, key=lambda i: (abs(active[i]), i))
        if abs(value) <= abs(active[smallest]):
            return False, None
        evicted = (smallest, active.pop(smallest))
        active[j] = value
        return True, evicted

    for t in range(40):
        x = rng.normal(size=9); x[rng.random(9) < .4] = 0.; x /= max(1., np.linalg.norm(x))
        present = np.flatnonzero(x); y = int(rng.choice([-1, 1]))
        pred = 0.
        for j in present:
            if j in active:
                value = active[j]
            elif method == 'awm':
                value = sum(signs[r, j] * S[r, buckets[r, j]] for r in range(3)) / 3
            else:
                value = 0.
            pred += value * x[j]
        eta = .1 / (1 + .1 * m.lam * t)
        old_members = set(active)
        S *= 1 - eta * m.lam
        active = {j: v * (1 - eta * m.lam) for j, v in active.items()}
        step = eta * (y - pred)
        if method == 'awm':
            for j in present:
                if j in old_members:
                    active[j] += step * x[j]
            for j in present:
                if j in old_members:
                    continue
                accepted, evicted = admit(j, query(j) + step * x[j])
                if not accepted:
                    add(j, step * x[j])
                elif evicted:
                    add(evicted[0], evicted[1] - query(evicted[0]))
        else:
            for j in present:
                add(j, step * x[j]); admit(j, query(j))
        assert m.update(k[present], x[present], y) == pytest.approx(pred, abs=1e-13)
        np.testing.assert_allclose(m.weights.reshape(3, 4) * m.scale, S, atol=1e-13, rtol=0)
        actual = {int(m.active.keys[s, 0]): m.active.values[s] * m.scale for s in range(m.active.n)}
        assert set(actual) == set(active)
        for j in actual:
            assert actual[j] == pytest.approx(active[j], abs=1e-13)
        m.active.check()


@pytest.mark.parametrize('method', ['hash_sgd', 'sparse_jl_sgd', 'awm', 'mission'])
@pytest.mark.parametrize('budget', [16*1024, 128*1024, 512*1024])
def test_budget_determinism_restart_and_readonly_prediction(method, budget):
    m = budgeted(method, budget, 100)
    rng = np.random.default_rng(42)
    for t in range(12):
        k = keys(np.arange(t * 20, t * 20 + 20)); x = rng.normal(size=20); x /= np.linalg.norm(x)
        m.update(k, x, 1 if t % 2 else -1)
        assert m.state_bytes() <= budget
    meta, arrays = m.snapshot()
    saved = {k: v.copy() for k, v in arrays.items()}
    restored = Learner.restore(json.loads(json.dumps(meta)), saved)
    for _ in range(5):
        assert m.predict(k, x) == restored.predict(k, x)
    assert m.snapshot()[0] == meta
    for key, a in m.snapshot()[1].items():
        np.testing.assert_array_equal(a, saved[key])
    for t in range(15):
        assert m.update(k, x, 1) == restored.update(k, x, 1)
    for key, a in m.snapshot()[1].items():
        np.testing.assert_array_equal(a, restored.snapshot()[1][key])
    assert m.state_bytes() <= budget


def test_inventory_counts_aliases_buffers_and_empty_capacity():
    a = np.zeros(100, np.float64)
    entries = inventory([('model', 'a', a), ('alias', 'again', a)])
    assert totals(entries)['payload_bytes'] == 800
    assert totals(entries)['bytes'] == sys.getsizeof(a)
    h = IndexedHeap(100)
    assert deep_bytes(h) > 100 * (16 + 8 + 4 + 4)


def test_causal_encoder_has_no_fitted_state_or_history():
    e = Encoder('B_causal')
    before = e.state_inventory()
    k, x = e.encode('A bank bank mortgage token-unseen-before')
    e.encode('tomorrow later label credit confidential future newwords')
    k2, x2 = e.encode('A bank bank mortgage token-unseen-before')
    np.testing.assert_array_equal(k, k2); np.testing.assert_array_equal(x, x2)
    assert before == e.state_inventory()
    assert e.vectorizer is None
    assert np.linalg.norm(x) == pytest.approx(1.)


def test_stream_order_budget_restart_memory_and_future_guard(tmp_path):
    path = tmp_path / 'train.jsonl'
    rows = [dict(id=str(i), date='2026-01-01', split='train', y=1, text='bank') for i in range(30)]
    write_spool(path, rows)
    a = Stream(path, 30, role='train')
    b = Stream(path, 30, role='train', order='random')
    assert b.order.nbytes == 30 * 4 and b.offsets.nbytes == 30 * 8
    assert b.state_bytes() >= a.state_bytes() + 30 * 12
    order = [r['id'] for r in b]
    assert len(set(order)) == 30 and order != [r['id'] for r in rows]
    b.close()
    b = Stream(path, 30, role='train', order='random', cursor=13)
    assert [r['id'] for r in b] == order[13:]
    a.close(); b.close()
    b = Stream(path, 30, role='train', cursor=13)
    assert [r['id'] for r in b] == [r['id'] for r in rows][13:]
    b.close()
    with pytest.raises(ValueError, match='Test access'):
        Stream(path, 30, role='test')
    rows[0]['date'] = '2026-02-01'; write_spool(path, rows)
    b = Stream(path, 30, role='train')
    with pytest.raises(ValueError, match='Future'):
        next(b)
    b.close()
