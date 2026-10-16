"""Sparse compression and vocabulary-free replay baselines; no Bank changes."""
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from .sketches import balanced_feature_bucket, sparse_jl_features
from .machine_size import qos_svm_machine_size

GRID = (500, 1000, 2000, 5000, 10000, 20000, 50000)
SEEDS = (42, 123, 456, 789, 2026)
ALPHAS = (1e-5, 1e-4, 1e-3)


def canonical32(matrix):
    result=sparse.csr_matrix(matrix,dtype=np.float32,copy=True)
    result.sum_duplicates();result.eliminate_zeros();result.sort_indices()
    return result


def rank_vocabulary(training_texts, vocabulary, batch_size=1024):
    """January corpus term frequency, descending; accepted column index ties.

    Feature indices themselves are alphabetical, not importance ranks. The
    supplied vocabulary has already passed January min_df/max_features. No
    vocabulary is fitted here and heldout texts must not be supplied.
    """
    counter=CountVectorizer(vocabulary={w:i for i,w in enumerate(vocabulary)},
        ngram_range=(1,2),stop_words='english',lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b')
    totals=np.zeros(len(vocabulary),dtype=np.int64)
    for start in range(0,len(training_texts),batch_size):
        totals+=np.asarray(counter.transform(training_texts[start:start+batch_size]).sum(axis=0)).ravel()
    return np.lexsort((np.arange(len(totals)),-totals)),totals


def compress(matrix, method, k, *, seed=42, ranking=None):
    """Explicit float32 CSR, identity copy, no input mutation.

    Truncation keeps frequency-ranked features, retains original feature order,
    and L2 renormalizes restricted TF-IDF. Sketch occupancy normalization is
    unchanged: no additional L2 normalization is applied to sketches.
    """
    if not sparse.isspmatrix_csr(matrix):raise TypeError('CSR input required')
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or k<1 or k>matrix.shape[1]:
        raise ValueError('k must be an integer within the input dimension')
    if method not in ['truncation','bucket','jl']:raise ValueError('Unknown method')
    if k==matrix.shape[1]:return canonical32(matrix)
    if method=='truncation':
        if ranking is None or len(ranking)!=matrix.shape[1] or set(ranking)!=set(range(matrix.shape[1])):
            raise ValueError('A training-derived feature permutation is required')
        return canonical32(normalize(matrix[:,np.sort(ranking[:k])],norm='l2',copy=True))
    result=(balanced_feature_bucket(matrix,k,seed=seed) if method=='bucket' else
            sparse_jl_features(matrix,k,seed=seed,sparsity=1))
    return canonical32(result)


def retention(score, reference):
    if not np.isfinite([score,reference]).all() or reference<=0:raise ValueError('Invalid score/reference')
    return dict(retention=float(score/reference),absolute_drop=float(reference-score))


def make_hasher(k):
    return HashingVectorizer(n_features=k,ngram_range=(1,2),alternate_sign=True,
        binary=False,norm='l2',dtype=np.float32,stop_words='english',lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b')


def fit_hash_stream(texts, labels, k, *, alphas=ALPHAS, passes=5, batch_size=1024, check=None):
    """Chronological replay: transform a batch, update each candidate sequentially.

    No fitted vocabulary/IDF or full training hash matrix. Labels are counted
    before five replay passes; not an unknown-frequency single-pass algorithm.
    Caller owns the text source/replay store and must account for it separately.
    """
    if passes<1 or batch_size<1 or len(texts)!=len(labels) or not len(texts):raise ValueError('Invalid stream')
    y=np.asarray(labels);counts=np.bincount(y,minlength=4)
    if len(counts)!=4 or (counts==0).any():raise ValueError('All four training classes required')
    weights=len(y)/(4*counts)
    models={a:SGDClassifier(loss='log_loss',penalty='l2',alpha=a,learning_rate='optimal',
             fit_intercept=True,average=False,random_state=42,shuffle=False) for a in alphas}
    fit_seconds={a:0. for a in alphas};hash_seconds=0.;nnz=0;row_max=0;columns=np.zeros(k,np.int64)
    max_batch=0;max_weights=0
    hasher=make_hasher(k)
    for epoch in range(passes):
        for start in range(0,len(texts),batch_size):
            if check:check()
            stop=min(start+batch_size,len(texts));t=time.perf_counter()
            x=canonical32(hasher.transform(texts[start:stop]));hash_seconds+=time.perf_counter()-t
            max_batch=max(max_batch,x.data.nbytes+x.indices.nbytes+x.indptr.nbytes)
            if epoch==0:
                nnz+=x.nnz;row_max=max(row_max,int(np.diff(x.indptr).max()))
                columns+=np.bincount(x.indices,minlength=k)
            sample_weights=weights[y[start:stop]].astype(np.float32);max_weights=max(max_weights,sample_weights.nbytes)
            for alpha,model in models.items():
                t=time.perf_counter();model.partial_fit(x,y[start:stop],classes=np.arange(4),sample_weight=sample_weights)
                fit_seconds[alpha]+=time.perf_counter()-t
    stats=dict(N=len(y),D=k,nnz=nnz,max_row_nnz=row_max,max_column_nnz=int(columns.max()),
               csr_bytes=nnz*8+(len(y)+1)*4,csr_bytes_kind='calculated payload if materialized; not stored',
               max_batch_csr_bytes=max_batch,sample_weight_buffer_bytes=max_weights,
               diagnostic_column_counter_bytes=columns.nbytes,training_class_counter_bytes=counts.nbytes,
               qos_row_only=qos_svm_machine_size(len(y),k,row_max),
               qos_row_or_column=qos_svm_machine_size(len(y),k,max(row_max,int(columns.max()))),
               passes=passes,batch_size=batch_size,hash_seconds_all_passes=hash_seconds,
               candidate_models_resident=len(models))
    return models,fit_seconds,stats


def predict_hash_stream(model,texts,k,*,batch_size=1024,check=None):
    hasher=make_hasher(k);pred=[];transform_time=0.;prediction_time=0.;max_batch=0
    for start in range(0,len(texts),batch_size):
        if check:check()
        t=time.perf_counter();x=canonical32(hasher.transform(texts[start:start+batch_size]));transform_time+=time.perf_counter()-t
        max_batch=max(max_batch,x.data.nbytes+x.indices.nbytes+x.indptr.nbytes)
        t=time.perf_counter();pred.append(model.predict(x));prediction_time+=time.perf_counter()-t
    return np.concatenate(pred),dict(transform_time=transform_time,prediction_time=prediction_time,
                                     max_prediction_batch_csr_bytes=max_batch)


def summarize_runs(raw):
    required={'method','k','seed','partition','population','macro_f1','balanced_accuracy',
              'accuracy','nnz','csr_bytes','parameter_bytes','qos_row_only','qos_row_or_column'}
    if not required<=set(raw):raise ValueError(f'Missing result columns: {required-set(raw)}')
    if raw.duplicated(['method','k','seed','partition','population']).any():raise ValueError('Duplicate run keys')
    numeric=[c for c in ['macro_f1','balanced_accuracy','accuracy','macro_f1_retention',
        'balanced_accuracy_retention','macro_f1_absolute_drop','balanced_accuracy_absolute_drop',
        'fit_time','training_transform_time','transform_time','prediction_time','peak_memory','nnz','csr_bytes','parameter_bytes',
        'qos_row_only','qos_row_or_column','max_row_nnz','max_column_nnz','zero_rows'] if c in raw]
    grouped=raw.groupby(['method','k','partition','population'],dropna=False,sort=True)
    agg=grouped[numeric].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns]
    return agg.join(grouped.size().rename('n_runs')).reset_index()
