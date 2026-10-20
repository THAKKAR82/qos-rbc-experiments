"""Independent implementations of declared, bounded, per-example updates.

Float64, half squared loss, no intercept, no class weighting, no averaging.
AWM follows the paper's projected-tail predictor. MISSION uses an exact bounded
touched-coordinate candidate heap; it does not enumerate global heavy hitters.
"""
import math
import numpy as np
from .state import inventory, totals

MASK = (1 << 64) - 1


def mix_scalar(x):
    x &= MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)


def mix_array(x):
    x = np.asarray(x, dtype=np.uint64).copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


class Projection:
    """Seed-generated signed block projection; no feature-indexed mapping."""
    __slots__ = ('seeds', 'width', 'depth')

    def __init__(self, width, depth, seed=42):
        self.width, self.depth = int(width), int(depth)
        if width < 1 or depth < 1:
            raise ValueError('Positive projection dimensions required')
        self.seeds = np.random.default_rng(np.random.SeedSequence([seed, 771])).integers(
            0, np.iinfo(np.uint64).max, size=(depth, 2), dtype=np.uint64)

    def hashes(self, keys):
        # Independent bucket/sign namespaces; finite keyed mixing is empirical,
        # not a claim of exact k-wise independence or a certified JL distortion.
        base = mix_array(keys[:, 0] ^ mix_array(keys[:, 1]))
        b = mix_array(base[None, :] ^ self.seeds[:, 0, None])
        s = mix_array(base[None, :] ^ self.seeds[:, 1, None])
        indices = (b % np.uint64(self.width)).astype(np.int64)
        indices += np.arange(self.depth)[:, None] * self.width
        signs = np.where(s & np.uint64(1), 1., -1.)
        return indices, signs


class IndexedHeap:
    """Fixed allocations: exact min-abs heap and linear-probe index.

    Backshift deletion prevents unbounded tombstones. Keys occupy two uint64s.
    Heap slots stay stable while heap positions change. Ties are deterministic.
    """
    __slots__ = ('keys', 'values', 'heap', 'positions', 'table', 'n', 'capacity')

    def __init__(self, capacity):
        self.capacity, self.n = int(capacity), 0
        self.keys = np.zeros((capacity, 2), np.uint64)
        self.values = np.zeros(capacity, np.float64)
        self.heap = np.zeros(capacity, np.int32)
        self.positions = np.zeros(capacity, np.int32)
        self.table = np.zeros(1 << max(2, (4 * capacity - 1).bit_length()), np.int32)

    def _home(self, lo, hi):
        return mix_scalar(lo ^ mix_scalar(hi)) & (len(self.table) - 1)

    def lookup(self, lo, hi):
        p = self._home(lo, hi)
        while self.table[p]:
            slot = int(self.table[p]) - 1
            if int(self.keys[slot, 0]) == lo and int(self.keys[slot, 1]) == hi:
                return slot
            p = (p + 1) & (len(self.table) - 1)
        return -1

    def _insert_index(self, slot):
        p = self._home(int(self.keys[slot, 0]), int(self.keys[slot, 1]))
        while self.table[p]:
            p = (p + 1) & (len(self.table) - 1)
        self.table[p] = slot + 1

    def _remove_index(self, slot):
        mask = len(self.table) - 1
        hole = self._home(int(self.keys[slot, 0]), int(self.keys[slot, 1]))
        while int(self.table[hole]) != slot + 1:
            hole = (hole + 1) & mask
        scan = (hole + 1) & mask
        while self.table[scan]:
            other = int(self.table[scan]) - 1
            home = self._home(int(self.keys[other, 0]), int(self.keys[other, 1]))
            if ((scan - home) & mask) >= ((hole - home) & mask):
                self.table[hole] = self.table[scan]
                hole = scan
            scan = (scan + 1) & mask
        self.table[hole] = 0

    def _less(self, a, b):
        va, vb = abs(self.values[a]), abs(self.values[b])
        return va < vb or (va == vb and (int(self.keys[a, 0]), int(self.keys[a, 1])) <
                          (int(self.keys[b, 0]), int(self.keys[b, 1])))

    def _swap(self, i, j):
        a, b = int(self.heap[i]), int(self.heap[j])
        self.heap[i], self.heap[j] = b, a
        self.positions[a], self.positions[b] = j, i

    def _repair(self, slot):
        p = int(self.positions[slot])
        while p and self._less(slot, int(self.heap[(p - 1) // 2])):
            parent = (p - 1) // 2
            self._swap(p, parent)
            p = parent
        while 2 * p + 1 < self.n:
            child = 2 * p + 1
            if child + 1 < self.n and self._less(int(self.heap[child + 1]), int(self.heap[child])):
                child += 1
            if not self._less(int(self.heap[child]), slot):
                break
            self._swap(p, child)
            p = child

    def set(self, slot, value):
        self.values[slot] = value
        self._repair(slot)

    def offer(self, lo, hi, value):
        """Return (accepted, evicted-key-and-value-or-None)."""
        slot = self.lookup(lo, hi)
        if slot >= 0:
            self.set(slot, value)
            return True, None
        evicted = None
        if self.n < self.capacity:
            slot = self.n
            self.heap[self.n] = slot
            self.positions[slot] = self.n
            self.n += 1
        else:
            slot = int(self.heap[0])
            if abs(value) <= abs(self.values[slot]):
                return False, None
            evicted = (int(self.keys[slot, 0]), int(self.keys[slot, 1]), float(self.values[slot]))
            self._remove_index(slot)
        self.keys[slot] = (lo, hi)
        self.values[slot] = value
        self._insert_index(slot)
        self._repair(slot)
        return True, evicted

    def check(self):
        assert self.n <= self.capacity
        assert np.count_nonzero(self.table) == self.n
        for pos in range(self.n):
            s = int(self.heap[pos])
            assert self.positions[s] == pos
            assert self.lookup(int(self.keys[s, 0]), int(self.keys[s, 1])) == s
            if pos:
                assert not self._less(s, int(self.heap[(pos - 1) // 2]))


class Learner:
    __slots__ = ('method', 'weights', 'projection', 'active', 'scale', 't', 'eta0', 'lam', 'budget',
                 'schedule', 'tau')
    SCHEDULES = ('inverse_time', 'inverse_sqrt')

    def __init__(self, method, width, *, depth=3, capacity=0, seed=42, eta=.1, lam=0., budget=2**63-1,
                 schedule='inverse_time', tau=None):
        if method not in ('hash_sgd', 'sparse_jl_sgd', 'awm', 'mission', 'identity_sgd'):
            raise ValueError('Unknown method')
        if method == 'mission' and lam != 0:
            raise ValueError('Native MISSION does not implement L2 regularization')
        if schedule not in self.SCHEDULES:
            raise ValueError('Unknown step-size schedule')
        if schedule == 'inverse_sqrt' and not (tau and tau > 0):
            raise ValueError('inverse_sqrt needs a positive time offset tau')
        self.schedule, self.tau = schedule, float(tau) if tau else 0.
        self.method, self.eta0, self.lam, self.budget = method, float(eta), float(lam), int(budget)
        self.scale, self.t = 1., 0
        depth = 1 if method in ('hash_sgd', 'identity_sgd') else depth
        self.projection = Projection(width, depth, seed)
        self.weights = np.zeros(width * depth, np.float64)
        self.active = IndexedHeap(capacity) if method in ('awm', 'mission') else None

    def state_inventory(self):
        # Split roots deliberately: active coefficients are model state, index
        # structures are active_index; shared objects are charged only once.
        roots = [('sketch' if self.active else 'model', 'weights', self.weights),
                 ('projection', 'projection', self.projection)]
        if self.active:
            roots.append(('model', 'active.values', self.active.values))
            roots.append(('active_index', 'active', self.active))
        roots.append(('optimizer', 'learner', self))
        return inventory(roots)

    def state_bytes(self):
        return totals(self.state_inventory())['bytes']

    def _features(self, keys, x):
        if self.method == 'identity_sgd':
            return keys[:, 0].astype(np.int64)[None, :], np.ones((1, len(x)))
        return self.projection.hashes(keys)

    def _prediction(self, keys, x, indices, signs):
        if self.active is None:
            return float(np.sum(self.weights[indices] * signs * x[None, :]) /
                         math.sqrt(self.projection.depth) * self.scale)
        slots = [self.active.lookup(int(lo), int(hi)) for lo, hi in keys]
        vals = np.array([self.active.values[s] if s >= 0 else 0. for s in slots])
        if self.method == 'awm':
            tail = np.asarray(slots) < 0
            vals[tail] = np.mean(self.weights[indices[:, tail]] * signs[:, tail], axis=0)
        return float(vals @ x * self.scale)

    def predict(self, keys, x):
        indices, signs = self._features(keys, x)
        return self._prediction(keys, x, indices, signs)

    def step_size(self, t):
        """Pilot default eta0/(1+eta0*lam*t); optional eta0/sqrt(1+t/tau)."""
        if self.schedule == 'inverse_time':
            return self.eta0 / (1 + self.eta0 * self.lam * t)
        return self.eta0 / math.sqrt(1 + t / self.tau)

    def update(self, keys, x, y):
        if y not in (-1, 1):
            raise ValueError('Binary ±1 label required')
        indices, signs = self._features(keys, x)
        pred = self._prediction(keys, x, indices, signs)
        eta = self.step_size(self.t)
        decay = 1 - eta * self.lam
        if decay <= 0:
            raise ValueError('Nonpositive lazy decay')
        self.scale *= decay
        if self.scale < 1e-100:
            self.weights *= self.scale
            if self.active:
                self.active.values *= self.scale
            self.scale = 1.
        gradient_step = eta * (y - pred) / self.scale
        if not math.isfinite(gradient_step):
            raise FloatingPointError('Nonfinite SGD update; no clipping applied')
        if self.active is None:
            np.add.at(self.weights, indices.ravel(),
                      (signs * x[None, :] * gradient_step / math.sqrt(self.projection.depth)).ravel())
        elif self.method == 'mission':
            for j, (lo, hi) in enumerate(keys):
                ix, sg = indices[:, j], signs[:, j]
                self.weights[ix] += sg * gradient_step * x[j]
                value = float(np.median(self.weights[ix] * sg))
                self.active.offer(int(lo), int(hi), value)
        else:
            # Freeze pre-update membership, update all active coordinates first.
            slots = [self.active.lookup(int(lo), int(hi)) for lo, hi in keys]
            for j, slot in enumerate(slots):
                if slot >= 0:
                    self.active.set(slot, self.active.values[slot] + gradient_step * x[j])
            for j, slot in enumerate(slots):
                if slot >= 0:
                    continue
                ix, sg = indices[:, j], signs[:, j]
                value = float(np.median(self.weights[ix] * sg)) + gradient_step * x[j]
                accepted, evicted = self.active.offer(int(keys[j, 0]), int(keys[j, 1]), value)
                if not accepted:
                    self.weights[ix] += sg * gradient_step * x[j]
                elif evicted is not None:
                    lo, hi, oldvalue = evicted
                    ei, es = self.projection.hashes(np.array([[lo, hi]], np.uint64))
                    ei, es = ei[:, 0], es[:, 0]
                    correction = oldvalue - float(np.median(self.weights[ei] * es))
                    self.weights[ei] += es * correction
        self.t += 1
        if not np.isfinite(self.weights[indices]).all():
            raise FloatingPointError('Nonfinite maintained state')
        return pred

    def snapshot(self):
        arrays = {'weights': self.weights, 'projection_seeds': self.projection.seeds}
        if self.active:
            arrays.update({'active_' + k: getattr(self.active, k)
                           for k in ('keys', 'values', 'heap', 'positions', 'table')})
        meta = dict(method=self.method, width=self.projection.width, depth=self.projection.depth,
                    capacity=self.active.capacity if self.active else 0,
                    active_n=self.active.n if self.active else 0, scale=self.scale, t=self.t,
                    eta=self.eta0, lam=self.lam, budget=self.budget, schedule=self.schedule, tau=self.tau)
        return meta, arrays

    @classmethod
    def restore(cls, meta, arrays):
        obj = cls(meta['method'], meta['width'], depth=meta['depth'], capacity=meta['capacity'],
                  eta=meta['eta'], lam=meta['lam'], budget=meta['budget'],
                  schedule=meta.get('schedule', 'inverse_time'), tau=meta.get('tau') or None)
        obj.weights[:] = arrays['weights']
        obj.projection.seeds[:] = arrays['projection_seeds']
        obj.scale, obj.t = meta['scale'], meta['t']
        if obj.active:
            for k in ('keys', 'values', 'heap', 'positions', 'table'):
                getattr(obj.active, k)[:] = arrays['active_' + k]
            obj.active.n = meta['active_n']
            obj.active.check()
        return obj


def budgeted(method, budget, n_train, seed=42, *, eta=.1, schedule='inverse_time', tau=None):
    # Fixed pilot settings, selected before data-dependent outcomes. A margin
    # covers scalar object de-aliasing as t/n change; allocated buffers never grow.
    capacity = 0
    if method in ('awm', 'mission'):
        target = max(1, int((budget - 2048) * .45 / 48))
        capacity = 1 << (target.bit_length() - 1)
    depth = 1 if method == 'hash_sgd' else 3
    lam = 0. if method == 'mission' else 1. / n_train
    lo, hi = 1, budget // (8 * depth)
    best = None
    while lo <= hi:
        width = (lo + hi) // 2
        candidate = Learner(method, width, depth=depth, capacity=capacity,
                            seed=seed, lam=lam, budget=budget, eta=eta, schedule=schedule, tau=tau)
        if candidate.state_bytes() + 512 <= budget:
            best, lo = width, width + 1
        else:
            hi = width - 1
    if best is None:
        raise MemoryError('infeasible_state')
    return Learner(method, best, depth=depth, capacity=capacity, seed=seed, lam=lam, budget=budget,
                   eta=eta, schedule=schedule, tau=tau)
