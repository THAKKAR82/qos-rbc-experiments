"""Explicit sparse holdout baselines; independent of Bank's CV evaluation.

Selection consumes primary validation results only. Missing-class strata retain
four-label macro-F1, but four-class balanced accuracy is undefined.
"""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import LinearSVC


def make_baseline(model, *, class_weight=None, alpha=10., C=1.):
    """Small prespecified baseline family; no implicit CV or hyperparameter search."""
    common = dict(class_weight=class_weight, fit_intercept=True, random_state=42, tol=1e-4)
    if model == 'ridge':
        return RidgeClassifier(alpha=alpha, solver='auto', **common)
    if model == 'logistic':
        # SAGA supports sparse float32 multinomial fitting. l1_ratio=0 is L2
        # in the pinned sklearn API; do not set the deprecated penalty option.
        return LogisticRegression(C=C, solver='saga', l1_ratio=0., max_iter=5000, **common)
    if model == 'linear_svc':
        return LinearSVC(C=C, penalty='l2', loss='squared_hinge', dual='auto',
                         max_iter=20000, **common)
    raise ValueError(f'Unknown baseline: {model}')


def classification_metrics(y_true, y_pred, *, labels=(0, 1, 2, 3)):
    """Fixed-order metrics, per-class rows and confusion counts/proportions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError('Expected nonempty aligned one-dimensional labels')
    if not set(y_true).union(y_pred).issubset(labels):
        raise ValueError('Labels outside the frozen label order')
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    proportions = np.divide(counts, support[:, None],
                            out=np.full(counts.shape, np.nan), where=support[:, None] != 0)
    metrics = dict(macro_f1=float(f1.mean()), accuracy=float(accuracy_score(y_true, y_pred)),
                   balanced_accuracy=float(recall.mean()) if (support > 0).all() else None,
                   absent_classes=[int(label) for label, n in zip(labels, support) if n == 0])
    per_class = [dict(label=int(label), precision=float(p), recall=float(r), f1=float(f), support=int(n))
                 for label, p, r, f, n in zip(labels, precision, recall, f1, support)]
    return metrics, per_class, counts, proportions


def select_validation(results):
    """Validation macro-F1, balanced accuracy, smaller C, stable ID tie-break.

    Reject test/secondary input even when its scores would pick the same model.
    """
    if not results or any(r['partition'] != 'validation' or r['population'] != 'primary'
                          for r in results):
        raise ValueError('Selection requires primary validation rows only')
    if any(r.get('balanced_accuracy') is None or
           not np.isfinite([r['macro_f1'], r['balanced_accuracy']]).all() for r in results):
        raise ValueError('Selection metrics must be finite and four-class')
    return min(results, key=lambda r: (-r['macro_f1'], -r['balanced_accuracy'],
                                      r.get('C', 1.), r['configuration_id']))['configuration_id']


def require_sparse(matrix):
    """Validate, without coercing dtype/representation or modifying input."""
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype not in (np.float32, np.float64):
        raise TypeError('Expected float32 or explicitly controlled float64 CSR')
    if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
        raise ValueError('Expected canonical finite CSR')


def coefficient_payload(estimator):
    """Array payload only: coefficient/intercept bytes exclude Python overhead."""
    coef = getattr(estimator, 'coef_', np.empty(0))
    intercept = getattr(estimator, 'intercept_', np.empty(0))
    return dict(coefficient_bytes=coef.nbytes, intercept_bytes=intercept.nbytes,
                parameter_bytes=coef.nbytes + intercept.nbytes,
                coefficient_dtype=str(coef.dtype))
