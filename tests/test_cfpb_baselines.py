"""Holdout-stage contracts: train-only vocabulary, controls, no Bank changes."""
import hashlib
import json
from contextlib import contextmanager
import numpy as np
import pandas as pd
import pytest
from src.cfpb_baselines import build_representation, configurations
from src.text_preprocessing import make_tfidf_vectorizer, text_matrix_stats


def test_prespecified_models_and_small_selection_scope():
    specs=configurations()
    assert len({s['configuration_id'] for s in specs})==12
    svm=[s for s in specs if s['model']=='linear_svc' and s['class_weight']=='balanced']
    assert [s['C'] for s in svm]==[.1,1.,10.]
    assert [s['alpha'] for s in specs if s['representation']=='unigram']==[10.,1.]
    assert len([s for s in specs if s['model']=='logistic'])==2
    assert len([s for s in specs if s['representation']=='primary_float64'])==2


def test_rebuild_verifies_training_fit_and_rejects_changed_idf(tmp_path,monkeypatch):
    @contextmanager
    def measurement(*args):yield {}
    monkeypatch.setattr('src.cfpb_baselines.measured',measurement)
    rows=[]
    for split in ['train','validation','test']:
        for label,term in enumerate(['collection debt','card purchase','checking deposit','mortgage house']):
            for repeat in range(2):
                rows.append(dict(complaint_id=str(len(rows)),label=label,split=split,primary=True,
                                 natural_stream=split!='train',text=term+(' uniqueheldoutword' if split!='train' else '')))
    rows=pd.DataFrame(rows);before=rows.copy(deep=True);train=rows[rows.split=='train']
    vec=make_tfidf_vectorizer();vec.fit(train.text)
    pd.DataFrame({'token':vec.get_feature_names_out()}).to_csv(tmp_path/'vocabulary.csv',index=False)
    np.save(tmp_path/'idf.npy',vec.idf_)
    (tmp_path/'vectorizer_manifest.json').write_text(json.dumps({'primary':dict(vocabulary_file='vocabulary.csv',idf_file='idf.npy',training_ids_sha256=hashlib.sha256('\n'.join(train.complaint_id).encode()).hexdigest())}))
    stats=[]
    for split in ['train','validation','test']:
        for pop in (['primary'] if split=='train' else ['primary','natural_stream']):
            stats.append(dict(**text_matrix_stats(vec.transform(rows[rows.split==split].text)),configuration_id='primary',population=pop,partition=split))
    pd.DataFrame(stats).to_csv(tmp_path/'representation_stats.csv',index=False)
    parts,matrices,st=build_representation('primary',rows,tmp_path,[])
    assert matrices['train'].dtype==np.float32
    np.testing.assert_allclose(matrices['test'].data,matrices['train'].data)
    assert 'uniqueheldoutword' not in vec.vocabulary_
    pd.testing.assert_frame_equal(before,rows)
    parts,x64,st64=build_representation('primary_float64',rows,tmp_path,[])
    assert x64['train'].dtype==np.float64 and x64['train'].shape==matrices['train'].shape
    np.save(tmp_path/'idf.npy',vec.idf_+1)
    with pytest.raises(AssertionError):build_representation('primary',rows,tmp_path,[])
