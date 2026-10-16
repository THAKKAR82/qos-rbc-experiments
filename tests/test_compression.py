import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from src.compression import (rank_vocabulary,compress,retention,fit_hash_stream,
                             predict_hash_stream,summarize_runs)
from src.text_preprocessing import text_matrix_stats


def test_frequency_truncation_ties_not_lexical_and_no_heldout_fit():
    rank,totals=rank_vocabulary(['zebra zebra beta alpha','beta zebra'],['alpha','beta','zebra'])
    assert rank.tolist()==[2,1,0] and totals.tolist()==[1,2,3]
    ties,_=rank_vocabulary(['zebra beta alpha'],['alpha','beta','zebra'])
    assert ties.tolist()==[0,1,2]
    x=sparse.csr_matrix([[.2,.4,.8],[1.,0.,0.]],dtype=np.float32)
    original=x.copy();y=compress(x,'truncation',2,ranking=rank)
    assert y.shape==(2,2) and y[1].nnz==0
    assert np.linalg.norm(y[0].data)==pytest.approx(1.)
    assert (original!=x).nnz==0
    # Transformation of a different population cannot alter training ranking.
    compress(sparse.csr_matrix([[9.,0.,0.]],dtype=np.float32),'truncation',2,ranking=rank)
    assert rank.tolist()==[2,1,0]


@pytest.mark.parametrize('method',['truncation','bucket','jl'])
def test_endpoint_sparse_dtype_determinism_nonmutation(method):
    rng=np.random.default_rng(9);x=sparse.csr_matrix(rng.normal(size=(12,20)).astype(np.float32));before=x.copy()
    rank=np.arange(20)[::-1]
    end=compress(x,method,20,ranking=rank)
    np.testing.assert_array_equal(end.data,x.data);assert end is not x and end.dtype==np.float32
    y=compress(x,method,4,seed=42,ranking=rank);z=compress(x,method,4,seed=42,ranking=rank)
    assert sparse.isspmatrix_csr(y) and y.has_canonical_format and y.shape==(12,4)
    assert (y!=z).nnz==0 and (x!=before).nnz==0
    stats=text_matrix_stats(y)
    assert stats['nnz']==y.nnz and stats['csr_bytes']==y.data.nbytes+y.indices.nbytes+y.indptr.nbytes
    assert stats['qos_row_or_column']>=stats['qos_row_only']
    if method!='truncation':assert (y!=compress(x,method,4,seed=123)).nnz>0


def test_hashing_batch_replay_no_fit_and_no_mutation(monkeypatch):
    monkeypatch.setattr(HashingVectorizer,'fit',lambda *a,**k:pytest.fail('Must not fit hashing vocabulary'))
    original=HashingVectorizer.transform;sizes=[]
    def tracked(self,texts):sizes.append(len(texts));return original(self,texts)
    monkeypatch.setattr(HashingVectorizer,'transform',tracked)
    texts=['collection debt','credit purchase','bank deposit','mortgage house']*4
    labels=np.tile(np.arange(4),4);before=labels.copy();copy=list(texts)
    models,times,stats=fit_hash_stream(texts,labels,100,alphas=[.001],passes=2,batch_size=3)
    assert len(sizes)==12 and max(sizes)<=3 and stats['passes']==2
    assert stats['N']==16 and stats['nnz']>0 and stats['max_batch_csr_bytes']<stats['csr_bytes']
    assert not hasattr(HashingVectorizer(),'vocabulary_')
    other,_,_=fit_hash_stream(texts,labels,100,alphas=[.001],passes=2,batch_size=3)
    p,info=predict_hash_stream(models[.001],texts,100,batch_size=3)
    q,_=predict_hash_stream(other[.001],texts,100,batch_size=3)
    np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(labels,before);assert texts==copy
    assert info['transform_time']>=0 and info['prediction_time']>=0


def test_retention_schema_aggregation_and_singletons():
    assert retention(.8,1.)==dict(retention=.8,absolute_drop=pytest.approx(.2))
    with pytest.raises(ValueError):retention(.8,0)
    rows=[]
    for seed,score in [(42,.8),(123,1.)]:
        rows.append(dict(method='jl',k=2,seed=seed,partition='validation',population='primary',
            macro_f1=score,balanced_accuracy=score,accuracy=score,nnz=9,csr_bytes=100,parameter_bytes=32,
            qos_row_only=12,qos_row_or_column=13))
    summary=summarize_runs(pd.DataFrame(rows));assert summary.iloc[0].macro_f1_mean==pytest.approx(.9)
    assert summary.iloc[0].macro_f1_std==pytest.approx(np.std([.8,1.],ddof=1))
    assert np.isnan(summarize_runs(pd.DataFrame(rows[:1])).iloc[0].macro_f1_std)
    with pytest.raises(ValueError):summarize_runs(pd.DataFrame(rows).drop(columns='nnz'))
    with pytest.raises(ValueError):summarize_runs(pd.DataFrame(rows+rows))
