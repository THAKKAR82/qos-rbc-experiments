"""Independent mathematical checks for the isolated fixed-test calibration."""
import numpy as np
import pytest
from numpy.polynomial import Chebyshev
from scipy.stats import binom

from qostight.phase_b_prototype import (augmented_selector, dense_stage,
                                       halmos, instance, shots_for_signal)
from qostight.qsvt_tools import verified_qsvt_angles


@pytest.mark.parametrize('N,s,kappa', [(4,2,2.),(8,4,4.),(64,16,16.),(128,2,8.)])
def test_controlled_instances_meet_measured_specs(N,s,kappa):
    X,y,x,w,m=instance(N,s,kappa,.1)
    assert np.linalg.norm(X,2)==pytest.approx(1)
    assert np.all(np.count_nonzero(X,axis=0)==s)
    assert np.all(np.count_nonzero(X,axis=1)==s)
    assert np.linalg.svd(X,compute_uv=False)[-1]<1e-12
    assert np.linalg.cond(X.T@X+m['lambda_reg']*np.eye(N))==pytest.approx(kappa**2)
    assert abs(x@w)/np.linalg.norm(w)==pytest.approx(.1)
    assert np.linalg.norm(x)==pytest.approx(1)
    assert set(y)=={-1,1} and m['r_reg']<=.99


def test_infeasible_margin_is_not_silently_changed():
    with pytest.raises(ValueError,match='infeasible'):
        instance(8,2,2.,1.1)


def test_augmented_block_preserves_factor_two():
    X,y,x,w,m=instance(4,2,2.,.1)
    B,checks=augmented_selector(X/2,m['lambda_reg'])
    b=np.r_[y/np.linalg.norm(y),np.zeros(4)]
    solved=np.linalg.lstsq(B,b,rcond=None)[0]
    expected=2*np.sqrt(1+m['lambda_reg'])*w/np.linalg.norm(y)
    assert np.allclose(solved,expected,atol=1e-12)
    assert checks['selector_unitarity']<1e-12
    assert checks['dirty_selector_difference']>.1


def test_complex_dense_lcu_and_dirty_ancilla_negative_control():
    B=np.array([[.1+.2j,.2],[.05j,-.2+.1j]])
    p=Chebyshev([0,.4,0,.1])
    angles,_=verified_qsvt_angles(p)
    r=dense_stage(B,p,np.asarray(angles))
    # Independent polynomial identity for odd singular-value transformation:
    # .4 T1 + .1 T3 = .1 B + .4 B B† B.
    expected=.1*B+.4*B@B.conj().T@B
    assert np.allclose(r['actual'],expected,atol=1e-10)
    assert r['lcu_unitarity']<1e-10
    assert r['dirty_ancilla_difference']>1e-3


def test_readout_shot_count_is_minimum_odd_binomial_budget():
    shots,p=shots_for_signal(.1,.05)
    assert shots%2==1 and p>=.95
    assert binom.sf((shots-2)//2,shots-2,.55)<.95
    with pytest.raises(ValueError):shots_for_signal(0,.1)


def test_noncontraction_rejected():
    with pytest.raises(ValueError,match='not a contraction'):halmos(np.eye(2)*1.1)


def test_degree_cap_fails_before_phase_finding():
    from qostight.phase_b_prototype import odd_interpolant
    with pytest.raises(RuntimeError,match='exceeds prototype cap'):
        odd_interpolant(lambda x:x,5001,1e-4,.1,lambda x:x)
