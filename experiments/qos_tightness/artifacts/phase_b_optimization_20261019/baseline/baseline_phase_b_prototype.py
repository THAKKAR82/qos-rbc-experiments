"""Tiny fixed-test Phase-(b) calibration; NOT a full F61/shadow/sampling sweep.

Stagewise canonical Halmos completion is an explicit reference-model assumption.
No stochastic oracle calls or fault-tolerant gate execution are implied.
"""
import math
import time
from functools import lru_cache

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.linalg import block_diag, sqrtm
from scipy.special import erfcinv, erf
from scipy.stats import beta, binom

from .env import load_reference
from .qsvt_tools import singular_value_transform, verified_qsvt_angles


def instance(N, s, kappa, gamma, seed=42):
    """Signed paired-difference circulant; exact null direction and uniform s.

    Correlated random signs preserve a known null singular value. New family,
    not the phase-(a) generator. Test norm is fixed at one; margin varies by angle.
    """
    if N < 4 or N & (N-1) or s % 2 or not 2 <= s <= N or kappa <= 1:
        raise ValueError('need power-of-two N>=4, even 2<=s<=N, kappa>1')
    rng = np.random.default_rng(seed)
    offsets = 2*rng.choice(N//2, s//2, replace=False)
    coeff = rng.uniform(0.5, 1.0, s//2)
    X = np.zeros((N, N))
    for offset, c in zip(offsets, coeff):
        X[np.arange(N), (np.arange(N)+offset) % N] = c
        X[np.arange(N), (np.arange(N)+offset+1) % N] = -c
    X *= rng.choice([-1., 1.], N)[:, None]
    X *= rng.choice([-1., 1.], N)[None, :]
    X /= np.linalg.norm(X, 2)
    # Teacher-generated binary labels; not an adversarial/worst-case label family.
    y = np.where(X @ rng.normal(size=N) >= 0, 1., -1.)
    lam = 1/(kappa*kappa-1)
    w = np.linalg.solve(X.T@X+lam*np.eye(N), X.T@y)
    wn = w/np.linalg.norm(w)
    support = np.argsort(-np.abs(wn), kind='stable')[:s]
    u = wn[support]; q = np.linalg.norm(u); u = u/q
    if not 0 < gamma < q:
        raise ValueError(f'infeasible fixed-norm margin {gamma}; support norm {q}')
    v = rng.normal(size=s); v -= u*np.dot(u, v); v /= np.linalg.norm(v)
    x = np.zeros(N); x[support] = gamma/q*u + np.sqrt(1-(gamma/q)**2)*v
    singulars = np.linalg.svd(X, compute_uv=False)
    measured_kappa = np.sqrt((singulars[0]**2+lam)/(singulars[-1]**2+lam))
    residual = (np.linalg.norm(X@w-y)**2+lam*np.linalg.norm(w)**2)/np.linalg.norm(y)**2
    if residual > .99:
        raise ValueError(f'nontrivial-residual gate failed: {residual}')
    assert np.all(np.count_nonzero(X, axis=0) == s)
    assert np.all(np.count_nonzero(X, axis=1) == s)
    assert abs(measured_kappa-kappa) < 1e-10
    assert abs(abs(x@wn)-gamma) < 1e-12 and np.max(np.abs(x)) <= 1
    return X, y, x, w, dict(N=N, D=N, seed=seed, s_row=s, s_col=s, s=s,
                            lambda_reg=lam, kappa_target=kappa, kappa_reg=measured_kappa,
                            gamma_target=gamma, gamma=float(abs(x@wn)), test_norm=float(np.linalg.norm(x)),
                            test_sparsity=int(np.count_nonzero(x)), test_infinity_norm=float(np.max(np.abs(x))),
                            singular_values=singulars.tolist(), r_reg=residual)


def odd_interpolant(target, start, eps, domain, desired, max_degree=4095):
    grid = np.linspace(-1, 1, 65537)
    check = np.unique(np.r_[np.linspace(domain, 1, 32769), domain])
    degree = int(start) | 1
    if degree > max_degree:
        raise RuntimeError(f'starting degree {degree} exceeds prototype cap {max_degree}')
    while degree <= max_degree:
        p = Chebyshev.interpolate(target, degree)
        coefs = p.coef.copy(); coefs[::2] = 0; p = Chebyshev(coefs)
        bound = float(np.max(np.abs(p(grid))))
        error = float(np.max(np.abs(p(check)-desired(check))))
        if bound < 1 and error <= eps:
            return p, dict(degree=int(p.degree()), max_abs_grid=bound, interval_error=error,
                           interval_min=domain, tolerance=eps, grid_points=len(grid))
        degree = (int(math.ceil(degree*1.4)) | 1)
    raise RuntimeError(f'no bounded polynomial by degree {max_degree}: error={error}, peak={bound}')


@lru_cache(maxsize=16)
def inverse_polynomial(kappa, eps_state):
    # Physical augmented block is A_lambda/2. K=2*kappa includes subnormalization.
    K = 2*kappa; eta = eps_state/(80*kappa)
    def target(x):
        u = K*x
        # C2 splice; bounded extension, not a claim of proof-optimal degree.
        inner = .5*(3*u-3*u**3+u**5)
        outer = np.divide(.5, u, out=np.zeros_like(u), where=u != 0)
        return np.where(np.abs(u) <= 1, inner, outer)
    p, info = odd_interpolant(target, int(16*K), eta, 1/K, lambda x: 1/(2*K*x))
    angles, chk = verified_qsvt_angles(p)
    return p, np.asarray(angles), dict(info, angle_error=chk['max_dev'], K=K,
                                      extension='C2 splice; degree not asserted proof-optimal')


@lru_cache(maxsize=16)
def fixed_point_polynomial(kappa, eps_state):
    threshold = 1/(40*kappa)
    # Amplitude loss must be O(eps_state^2) to bound FULL state error, including garbage.
    tol = eps_state**2/128
    k = erfcinv(tol/4)/threshold
    target = lambda x: (1-tol/4)*erf(k*x)
    p, info = odd_interpolant(target, int(2*k)+1, tol, threshold, lambda x: np.ones_like(x))
    angles, chk = verified_qsvt_angles(p)
    return p, np.asarray(angles), dict(info, angle_error=chk['max_dev'],
                                      threshold_policy='theorem 1/(40*kappa)',
                                      full_state_error_bound=float(np.sqrt(2*tol)))


def halmos(B):
    B = np.asarray(B, complex)
    m, n = B.shape
    if np.linalg.norm(B, 2) > 1+1e-10:
        raise ValueError('not a contraction')
    return np.block([[B, np.asarray(sqrtm(np.eye(m)-B@B.conj().T), complex)],
                     [np.asarray(sqrtm(np.eye(n)-B.conj().T@B), complex), -B.conj().T]])


def dense_stage(B, p, angles):
    """Explicit Hermitian embedding + Halmos + reference QSVT + real-part LCU.

    Pad to qubit dimensions. Compare complex rectangular success block, without
    discarding imaginary information from complex input states.
    """
    ref = load_reference()
    import jax.numpy as jnp
    m, n = B.shape; dim = 1 << (m+n-1).bit_length()
    H = np.zeros((dim, dim), complex)
    H[:m, m:m+n] = B; H[m:m+n, :m] = B.conj().T
    U = halmos(H)
    V = np.asarray(ref.qsvt.apply_qsvt(jnp.asarray(U), 1, jnp.asarray(angles)))
    # (H_control tensor I) diag(V,V†) (H_control tensor I)
    R = (V+V.conj().T)/2; S = (V-V.conj().T)/2
    lcu = np.block([[R, S], [S, R]])
    actual = lcu[:m, m:m+n]
    expected = singular_value_transform(B, p)
    dirty = lcu[:m, 2*dim+m:2*dim+m+n]
    return dict(max_abs_diff=float(np.max(np.abs(actual-expected))),
                qsvt_unitarity=float(np.linalg.norm(V.conj().T@V-np.eye(len(V)), 2)),
                lcu_unitarity=float(np.linalg.norm(lcu.conj().T@lcu-np.eye(len(lcu)), 2)),
                dirty_ancilla_difference=float(np.linalg.norm(dirty-expected)),
                physical_dense_dimension=len(lcu), actual=actual)


def augmented_selector(C, lam):
    """One selector rotation, controlled block encodings of C=X/2 and I/2.

    Keep selector output as augmented row index. Both input selector and BE
    ancilla must be zero. No multiplication of the success block by two.
    """
    N = len(C); a=1/np.sqrt(1+lam); b=np.sqrt(lam)/np.sqrt(1+lam)
    rotation = np.array([[a, -b], [b, a]])
    U = block_diag(halmos(C), halmos(np.eye(N)/2)) @ np.kron(rotation, np.eye(2*N))
    indices = np.r_[np.arange(N), np.arange(2*N, 3*N)]
    block = U[indices, :N]
    expected = np.vstack([C, np.sqrt(lam)*np.eye(N)/2])/np.sqrt(1+lam)
    return block, dict(selector_block_error=float(np.max(np.abs(block-expected))),
                       selector_unitarity=float(np.linalg.norm(U.conj().T@U-np.eye(4*N), 2)),
                       dirty_selector_difference=float(np.linalg.norm(U[indices, 2*N:3*N]-expected)))


def wilson_or_exact(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k<n else 1.]


def shots_for_signal(signal, delta):
    if not 0 < signal < 1 or not 0 < delta < 1:
        raise ValueError('need nonzero signal and 0<delta<1')
    def success(shots): return float(binom.sf(shots//2, shots, (1+signal)/2))
    high=1
    while success(high)<1-delta:
        high=2*high+1
    lo,hi=0,(high-1)//2
    while lo<hi:
        mid=(lo+hi)//2
        if success(2*mid+1)>=1-delta:hi=mid
        else:lo=mid+1
    shots=2*lo+1
    return shots, success(shots)


def query_accounting(N, s, M, degrees, shots):
    """Calculated conservative recipe, NOT a traced nested sampled circuit.

    At each new QSVT stage: two controlled Hermitianization branches and two
    real-part LCU branches. Same samples may serve paired conjugate branches.
    This upper-bound recipe requires a literal nested-circuit audit before sweeps.
    """
    ds, da, di, df = (degrees[k] for k in ('sign','d23','inverse','fixed_point'))
    bits=int(math.log2(N))
    sparse_blocks=2*bits*ds+2  # row, col, element and its adjoint, phase-(a) convention
    # Explicit phase-oracle sine LCU + sign real-part LCU applications.
    sparse_phase_branches=8*bits*ds+4
    x_calls=16*df*di; y_calls=4*df
    elementary_branches=x_calls*4*da*sparse_phase_branches+y_calls
    fresh_blocks=df*(di*da*sparse_blocks+1)
    return dict(accounting_status='calculated conservative branch recipe; nested trace not yet validated',
                lcu_extra_clean_ancillas_per_stage=1,
                hermitianization_branch_factor=2, real_part_lcu_branch_factor=2,
                Q0_X_branches_per_shot=x_calls, Q0_y_branches_per_shot=y_calls,
                Q0_branch_sum_per_shot=x_calls+y_calls,
                d23_sparse_BE_branches_per_call=4*da,
                elementary_phase_branches_per_shot=elementary_branches,
                fresh_sample_blocks_per_shot_reference_sharing=fresh_blocks,
                samples_X_per_shot_reference_sharing=M*df*di*da*sparse_blocks,
                samples_y_per_shot_reference_sharing=M*df,
                total_samples_reference_sharing=shots*M*fresh_blocks,
                total_samples_no_conjugate_sharing=shots*M*elementary_branches,
                equivalent_row_passes_reference_sharing=shots*M*fresh_blocks/N,
                equivalent_entry_passes_X_reference_sharing=shots*M*df*di*da*sparse_blocks/(N*s),
                readout_shots=shots, samples_per_elementary_oracle=M)


def calibrate(N=4, s=2, kappa=2., gamma=.3, delta=.1, eps_state=.01, M=10**8, seed=42):
    from .blockenc import sketched_block_encoding, d23_amplify
    from .polynomials import amplification_polynomial
    timings={}
    def timed(name, fn):
        t=time.perf_counter();out=fn();timings[name]=time.perf_counter()-t;return out
    X,y,x,w,meta=instance(N,s,kappa,gamma,seed)
    lam=meta['lambda_reg']; wn=w/np.linalg.norm(w)
    # Primitive accuracy is an explicit prototype setting, not delta/(6Q).
    eps_be=1e-4
    be=timed('sparse_oracles',lambda: sketched_block_encoding(X,M,eps_target=eps_be))
    amp=timed('d23_shortcut',lambda:d23_amplify(be,eps=eps_be)); C=amp['block']
    pa=amplification_polynomial(s,eps=eps_be)
    aa,qa=timed('d23_angles',lambda:verified_qsvt_angles(pa))
    dc=timed('d23_dense',lambda:dense_stage(be['B'],pa,np.asarray(aa)))
    A,selector=augmented_selector(C,lam)
    ideal_A=np.vstack([X,np.sqrt(lam)*np.eye(N)])/np.sqrt(1+lam)
    b=np.r_[y/np.linalg.norm(y),np.zeros(N)]
    lstsq=np.linalg.lstsq(ideal_A,b,rcond=None)[0]
    ref=load_reference()
    import jax.numpy as jnp
    ysketch,_=ref.qos.q_state_sketch_flat(jnp.asarray(y),M)
    bs=np.r_[np.asarray(ysketch),np.zeros(N)]
    pinv,ai,pi=timed('inverse_polynomial_and_angles',lambda:inverse_polynomial(kappa,eps_state))
    T=timed('inverse_shortcut',lambda:singular_value_transform(A.conj().T,pinv))
    invdense=timed('inverse_dense',lambda:dense_stage(A.conj().T,pinv,ai))
    z=T@bs
    target_z=lstsq/(2*kappa)
    fp,af,pf=timed('fixed_point_polynomial_and_angles',lambda:fixed_point_polynomial(kappa,eps_state))
    a=float(np.linalg.norm(z))
    if a < pf['interval_min'] or a>1:
        raise RuntimeError('amplification success amplitude outside verified interval')
    good=timed('fixed_point_shortcut',lambda:singular_value_transform(z[:,None],fp)[:,0])
    fpdense=timed('fixed_point_dense',lambda:dense_stage(z[:,None],fp,af))
    goodnorm=float(np.linalg.norm(good)); leakage=math.sqrt(max(0,1-goodnorm**2))
    full_state_error=math.sqrt(float(np.linalg.norm(good-wn)**2)+leakage**2)
    # Full states include garbage orthogonal to the clean test branch; no postselection.
    test=np.r_[x,np.zeros(N)]; output=np.r_[good,np.r_[leakage,np.zeros(N-1)]]
    psi=np.r_[test,output]/np.sqrt(2)
    after=(psi[:2*N]+psi[2*N:])/np.sqrt(2)
    pzero=float(np.vdot(after,after).real)
    mean=float(np.vdot(x,good).real)
    shots,p_correct=shots_for_signal(abs(mean),delta)
    draws=np.random.default_rng(np.random.SeedSequence([seed,20261019])).binomial(shots,pzero,size=200)
    predicted=np.where(draws>shots//2,1,-1); target_label=1 if x@w>0 else -1
    successes=int(np.count_nonzero(predicted==target_label))
    ideal_pzero=(1+float(x@wn))/2
    gates=dict(d23_dense_error=dc['max_abs_diff'], inverse_dense_error=invdense['max_abs_diff'],
               fixed_point_dense_error=fpdense['max_abs_diff'],
               augmented_gram_error=float(np.linalg.norm(ideal_A.T@ideal_A-(X.T@X+lam*np.eye(N))/(1+lam))),
               augmented_lstsq_vs_ridge=float(np.linalg.norm(lstsq/np.linalg.norm(lstsq)-wn)),
               d23_vs_X_half=float(np.linalg.norm(C-X/2,2)),
               inverse_success_vs_exact=float(np.linalg.norm(z-target_z)),
               label_state_error=float(np.linalg.norm(bs-b)),
               full_state_error=full_state_error, leakage_amplitude=leakage,
               readout_formula_error=abs(pzero-(1+mean)/2),
               readout_vs_ideal=abs(pzero-ideal_pzero),
               label_matches_exact=bool(np.sign(mean)==target_label))
    for name,r in [('d23',dc),('inverse',invdense),('fixed_point',fpdense)]:
        gates[name+'_unitarity']=r['lcu_unitarity']
        gates[name+'_dirty_ancilla_difference']=r['dirty_ancilla_difference']
    gates.update(selector)
    if max(dc['max_abs_diff'],invdense['max_abs_diff'],fpdense['max_abs_diff'])>1e-9:
        raise RuntimeError(f'dense QSVT gate failed: {gates}')
    if full_state_error>eps_state or not gates['label_matches_exact']:
        raise RuntimeError(f'full-state correctness gate failed: {gates}')
    degrees=dict(sign=be['sign_degree'],d23=int(pa.degree()),inverse=int(pinv.degree()),fixed_point=int(fp.degree()))
    resources=query_accounting(N,s,M,degrees,shots)
    return dict(**meta,delta=delta,eps_target=eps_state,eps_BE=eps_be,M=M,degrees=degrees,
                inverse=pi,fixed_point=pf,d23_angle_error=qa['max_dev'],gates=gates,
                success_amplitude_before_fixed_point=a,success_norm_after_fixed_point=goodnorm,
                p_zero=pzero,ideal_p_zero=ideal_pzero,mean_signal=mean,
                classification_probability=p_correct,shot_repetitions=200,correct_repetitions=successes,
                correctness_ci95=wilson_or_exact(successes,200),resources=resources,timings=timings,
                interpretation='simulation (expected-unitary QOS, reference convention)',
                readout_model='fixed-test Hadamard; not reusable Clifford shadows',
                completion_model='stagewise canonical Halmos; changing random garbage not validated',
                sign_verified=all(v['verified'] for v in be['sign_info'].values()))
