"""Bookkeeping reserve regressions; synthetic states only, no CFPB fitting."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import methods as mm


BUDGETS = tuple(k * 1024 for k in (4, 16, 32, 64, 128, 512))
METHODS = ("hash_sgd", "sparse_jl_sgd", "awm", "mission")


def keys(ids):
    return np.column_stack((np.asarray(ids, np.uint64), np.zeros(len(ids), np.uint64)))


def saved_state(model):
    meta, arrays = model.snapshot()
    return meta, {key: value.copy() for key, value in arrays.items()}


def assert_saved_state(model, saved):
    meta, arrays = model.snapshot()
    assert meta == saved[0]
    assert arrays.keys() == saved[1].keys()
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, saved[1][name])


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schedule", ("inverse_time", "inverse_sqrt"))
@pytest.mark.parametrize("eta", (0.03, 0.1, 0.3, 1.0))
def test_every_approved_budget_initialized_and_trained(method, budget, schedule, eta):
    """Cross scalar-cache boundaries through real updates at every approved budget."""
    model = mm.budgeted(method, budget, 7265, seed=2026, eta=eta, schedule=schedule,
                        tau=7265 if schedule == "inverse_sqrt" else None)
    provenance = mm.allocation_metadata(model)
    bound = mm.allocation_bound(model)
    assert provenance["version"] == mm.ALLOCATOR_VERSION
    assert provenance["initialized_state_bytes"] == model.state_bytes()
    assert provenance["state_bound_bytes"] == bound
    assert provenance["bookkeeping_reserve_bytes"] == bound - model.state_bytes()
    assert provenance["margin_bytes"] == mm.STATE_MARGIN_BYTES == 512
    assert provenance["max_updates"] == mm.MAX_UPDATES == 2**63 - 1
    assert provenance["width"] == model.projection.width
    assert provenance["depth"] == model.projection.depth
    assert provenance["capacity"] == (model.active.capacity if model.active else 0)
    assert model.state_bytes() <= bound
    assert bound + mm.STATE_MARGIN_BYTES <= budget
    k, x = keys([11, 19]), np.array([0.6, 0.8])
    for i in range(258):
        model.update(k, x, 1 if i % 2 else -1)
        if i in (0, 1, 2, 255, 256, 257):
            assert model.state_bytes() <= bound
            assert model.state_bytes() + mm.STATE_MARGIN_BYTES <= budget
            assert mm.allocation_bound(model) == bound
    assert model.t == 258
    if model.active:
        model.active.check()


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("method", METHODS)
def test_supported_counter_sizes_and_heap_occupancies_fit_bound(method, budget):
    """Synthetic scalar probes cover huge counters without billions of updates."""
    model = mm.budgeted(method, budget, 7265)
    bound = mm.allocation_bound(model)
    probe = copy.copy(model)
    if model.active:
        probe.active = copy.copy(model.active)
    # Only scalar accounting is inspected: these probes do not claim to be
    # operational heap snapshots and are never updated/predicted/restored.
    occupancies = (0, 1, 7, model.active.capacity) if model.active else (None,)
    for counter in (0, 1, 257, 2**30 - 1, 2**30, mm.MAX_UPDATES):
        probe.t = int(str(counter))
        probe.scale = float("0.125")
        for occupancy in occupancies:
            if probe.active:
                probe.active.n = int(str(occupancy))
            assert probe.state_bytes() <= bound
            assert probe.state_bytes() + mm.STATE_MARGIN_BYTES <= budget
            assert mm.allocation_bound(probe) == bound
    assert model.t == 0
    if model.active:
        assert model.active.n == 0


@pytest.mark.parametrize("method", METHODS)
def test_bound_and_metadata_preserve_state_rng_and_fixed_dimension_updates(method):
    model = mm.budgeted(method, 65536, 7265, eta=1.0)
    meta, arrays = saved_state(model)
    twin = mm.Learner.restore(meta, arrays)
    initial_bytes = model.state_bytes()
    original_ids = (id(model), id(model.active), id(model.t), id(model.scale))
    random_state = np.random.get_state()
    for _ in range(3):
        mm.allocation_bound(model)
        mm.allocation_metadata(model)
    assert_saved_state(model, (meta, arrays))
    assert model.state_bytes() == initial_bytes
    assert original_ids == (id(model), id(model.active), id(model.t), id(model.scale))
    after = np.random.get_state()
    assert random_state[0] == after[0]
    np.testing.assert_array_equal(random_state[1], after[1])
    assert random_state[2:] == after[2:]
    # Interleaving accounting with updates must preserve predictions and exact
    # coefficient/heap state for an identical, fixed-dimensional learner.
    for i in range(20):
        k = keys([i, i + 1, i + 5])
        x = np.array([0.2, 0.3, 0.4])
        mm.allocation_bound(model)
        assert model.update(k, x, 1 if i % 2 else -1) == twin.update(k, x, 1 if i % 2 else -1)
        assert_saved_state(model, saved_state(twin))


def test_bound_accounts_for_immutable_aliases_and_preserves_them_on_error(monkeypatch):
    model = mm.Learner("awm", 7, capacity=16, eta=1.0, lam=0.0)
    # Explicit aliases avoid depending on compiler float-constant interning.
    model.scale = model.eta0
    model.tau = model.lam
    model.t = model.active.n = model.projection.depth
    before = saved_state(model)
    identities = (id(model.scale), id(model.eta0), id(model.tau), id(model.lam),
                  id(model.t), id(model.active.n), id(model.projection.depth))
    bound = mm.allocation_bound(model)
    assert_saved_state(model, before)
    probe = copy.copy(model)
    probe.active = copy.copy(model.active)
    probe.scale = float("0.125")
    probe.t = mm.MAX_UPDATES
    probe.active.n = 7
    assert probe.state_bytes() <= bound

    def fail_inventory(*args, **kwargs):
        raise RuntimeError("synthetic accounting failure")

    monkeypatch.setattr(mm, "inventory", fail_inventory)
    with pytest.raises(RuntimeError, match="synthetic accounting failure"):
        mm.allocation_bound(model)
    assert_saved_state(model, before)
    assert identities == (id(model.scale), id(model.eta0), id(model.tau), id(model.lam),
                          id(model.t), id(model.active.n), id(model.projection.depth))


def test_known_legacy_64k_awm_state_growth_is_charged_by_bound():
    legacy_dimensions = mm.Learner("awm", 1624, depth=3, capacity=512, seed=2026,
                                   eta=0.1, lam=1 / 7265, budget=65536)
    initial = legacy_dimensions.state_bytes()
    legacy_dimensions.update(keys([0, 1, 2, 4, 8, 16, 32]), np.ones(7) / np.sqrt(7), 1)
    trained = legacy_dimensions.state_bytes()
    assert trained - initial == sys.getsizeof(0)
    assert trained <= mm.allocation_bound(legacy_dimensions)
    # Exact object sizes are interpreter/platform dependent. On the recorded
    # failure platform these reproduce the reported 17-byte final-check excess.
    if initial == 65013:
        assert trained == 65041
        assert initial + 512 <= 65536 < trained + 512
    corrected = mm.budgeted("awm", 65536, 7265, seed=2026)
    corrected.update(keys([0, 1, 2, 4, 8, 16, 32]), np.ones(7) / np.sqrt(7), 1)
    assert corrected.state_bytes() <= mm.allocation_bound(corrected)
    assert mm.allocation_bound(corrected) + 512 <= 65536


@pytest.mark.parametrize("method", METHODS)
def test_counter_limit_raises_before_any_persistent_mutation(method):
    model = mm.budgeted(method, 65536, 7265)
    model.t = mm.MAX_UPDATES - 1
    model.update(keys([0, 1]), np.array([0.6, 0.8]), 1)
    assert model.t == mm.MAX_UPDATES
    assert model.state_bytes() <= mm.allocation_bound(model)
    saved = saved_state(model)
    with pytest.raises(OverflowError):
        model.update(keys([0, 1]), np.array([0.6, 0.8]), -1)
    assert_saved_state(model, saved)


@pytest.mark.parametrize("slot,value", [("t", -1), ("t", mm.MAX_UPDATES + 1), ("n", -1), ("n", 17)])
def test_bound_rejects_unsupported_bookkeeping_states(slot, value):
    model = mm.Learner("awm", 7, capacity=16)
    setattr(model if slot == "t" else model.active, slot, value)
    with pytest.raises(ValueError, match="outside the declared allocation bound"):
        mm.allocation_bound(model)
