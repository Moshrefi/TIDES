import numpy as np
import torch

from tides import TIDES, TIDESConfig
from tides.eval.metrics import Events, MetricConfig, all_metrics, chamfer_st, same_ts_fraction
from tides.events.arbiter import ArbiterConfig, _fifo, apply_readout
from tides.events.comparator import ComparatorConfig, ResidualComparator
from tides.scene.synthetic import occlusion_scene, static_trajectory


def test_linear_ramp_gives_exact_crossing_times():
    # S(t) = 2 t on one pixel, C = 0.2 => events at t = 0.1, 0.2, ... (multi-crossing in one step).
    comp = ResidualComparator(1, ComparatorConfig(C_pos=0.2, C_neg=0.2, n_max=20, local_model="linear"),
                              dtype=torch.float64)
    comp.reset(torch.zeros(1, dtype=torch.float64))
    comp.step(0.0, 0.55, torch.zeros(1, dtype=torch.float64), torch.full((1,), 2.0, dtype=torch.float64))
    t, idx, p = comp.events()
    assert np.allclose(t.numpy(), [0.1, 0.2, 0.3, 0.4, 0.5])
    assert (p.numpy() == 1).all()


def test_quadratic_model_hits_curved_signal():
    # S(t) = t^2 over (0,1]; quadratic local model from S0=0, S0'=0, S1=1 is exact.
    comp = ResidualComparator(1, ComparatorConfig(C_pos=0.25, C_neg=0.25, n_max=10), dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)
    comp.reset(z)
    comp.step(0.0, 1.0, z, z, S1=torch.ones(1, dtype=torch.float64))
    t, _, _ = comp.events()
    assert np.allclose(t.numpy(), np.sqrt([0.25, 0.5, 0.75, 1.0]))


def test_negative_events_and_masking():
    comp = ResidualComparator(2, ComparatorConfig(C_pos=0.1, C_neg=0.1, n_max=5, local_model="linear"),
                              dtype=torch.float64)
    comp.reset(torch.zeros(2, dtype=torch.float64))
    comp.step(0.0, 1.0, torch.zeros(2, dtype=torch.float64), torch.tensor([-0.25, -0.25], dtype=torch.float64),
              emit_mask=torch.tensor([True, False]))
    t, idx, p = comp.events()
    assert np.allclose(t.numpy(), [0.4, 0.8]) and (idx.numpy() == 0).all() and (p.numpy() == -1).all()
    assert comp.suppressed == 2


def test_fifo_capacity():
    arrival = np.zeros(6)
    out = _fifo(arrival, np.zeros(6, dtype=np.int64), np.full(6, 2, dtype=np.int64), 1e-3)
    assert np.allclose(out, [0, 0, 1e-3, 1e-3, 2e-3, 2e-3])


def test_readout_drops_and_spreads_under_activity():
    rng = np.random.default_rng(0)
    n = 5000
    t = np.sort(rng.uniform(0, 1e-3, n))
    x = rng.integers(0, 32, n)
    y = rng.integers(0, 32, n)
    p = np.ones(n, dtype=np.int8)
    cfg = ArbiterConfig(enabled=True, eta=0.3, gamma_sigma=0.0, K0=2)
    t2, *_ = apply_readout(t, x, y, p, 32, np.ones((1, 1)), np.array([0.0]), cfg)
    assert len(t2) < n * 0.8 and t2.max() > t.max()


def test_metrics_identity_and_batching():
    rng = np.random.default_rng(0)
    n = 4000
    ev = Events(np.sort(rng.uniform(0, 0.1, n)), rng.integers(0, 64, n), rng.integers(0, 48, n),
                rng.choice([-1, 1], n).astype(np.int8))
    cfg = MetricConfig(window=0.02)
    assert chamfer_st(ev, ev, 64, 48, cfg) == 0.0
    batched = Events(np.round(ev.t, 2), ev.x, ev.y, ev.p)
    assert same_ts_fraction(batched, cfg) > 0.9 > same_ts_fraction(ev, cfg)
    m = all_metrics(ev, ev, 64, 48, cfg)
    assert m["Fano"] == 0.0


def test_simulator_end_to_end_no_batching():
    sc, tr = occlusion_scene(), static_trajectory(t1=0.2, n=5)
    cfg = TIDESConfig(device="cpu", queries_per_frame=4)
    cfg.comparator.C_pos = cfg.comparator.C_neg = 0.2
    out = TIDES(sc, tr, cfg).simulate(verbose=False)
    assert len(out["t"]) > 100
    assert np.all(np.diff(out["t"]) >= 0)
    # Continuous-time crossings: almost no two events share a timestamp.
    assert len(np.unique(out["t"])) / len(out["t"]) > 0.95
    # Finite-difference ablation runs too.
    cfg.slope = "finite_diff"
    cfg.adaptive = False
    out_fd = TIDES(sc, tr, cfg).simulate(verbose=False)
    assert len(out_fd["t"]) > 0
