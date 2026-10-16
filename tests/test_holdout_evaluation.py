import numpy as np
import pytest
from scipy import sparse
from src.holdout_evaluation import (make_baseline, classification_metrics,
                                    select_validation, require_sparse, coefficient_payload)

@pytest.mark.parametrize('model', ['ridge','logistic','linear_svc'])
def test_sparse_fit_configuration_determinism_and_nonmutation(model):
    x=sparse.csr_matrix(np.tile(np.eye(4,dtype=np.float32),(8,1)))
    y=np.tile(np.arange(4),8); before=(x.data.copy(),x.indices.copy(),x.indptr.copy(),y.copy())
    estimator=make_baseline(model,class_weight='balanced')
    assert estimator.random_state==42 and estimator.tol==1e-4 and estimator.fit_intercept
    estimator.fit(x,y); other=make_baseline(model,class_weight='balanced').fit(x,y)
    np.testing.assert_array_equal(estimator.predict(x),other.predict(x))
    for got,old in zip((x.data,x.indices,x.indptr,y),before):np.testing.assert_array_equal(got,old)
    assert coefficient_payload(estimator)['parameter_bytes']>0
    require_sparse(x)
    if model=='ridge':assert estimator.alpha==10 and estimator.solver=='auto'
    if model=='logistic':assert estimator.solver=='saga' and estimator.C==1 and estimator.l1_ratio==0
    if model=='linear_svc':assert estimator.dual=='auto' and estimator.max_iter==20000

def test_metrics_confusion_and_missing_class_contract():
    metrics,rows,cm,norm=classification_metrics([0,0,1,2,3],[0,1,1,2,2])
    assert metrics['accuracy']==.6 and metrics['balanced_accuracy']==.625
    assert cm.sum()==5 and cm[0,1]==1 and norm[0,1]==.5
    assert rows[3]['recall']==0
    metrics,rows,cm,norm=classification_metrics([0,1],[0,1])
    assert metrics['macro_f1']==.5 and metrics['balanced_accuracy'] is None
    assert metrics['absent_classes']==[2,3] and np.isnan(norm[2:]).all()

def test_selection_ties_and_heldout_rejection():
    def row(c):return dict(configuration_id=str(c),C=c,macro_f1=.8,balanced_accuracy=.7,
                          population='primary',partition='validation')
    assert select_validation([row(10),row(.1),row(1)])=='0.1'
    for field,value in [('partition','test'),('population','natural_stream')]:
        r=row(1);r[field]=value
        with pytest.raises(ValueError):select_validation([r])

def test_representation_and_labels_rejected_without_silent_conversion():
    with pytest.raises(TypeError):require_sparse(np.eye(4))
    with pytest.raises(ValueError):classification_metrics([4],[0])
