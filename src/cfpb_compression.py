"""CFPB compression stage; python -m src.cfpb_compression. Fresh outputs only."""
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .cfpb import CLASSES,SOURCE_SHA256
from .cfpb_prepare import ROOT,sha256_file,write_json,git_commit,check_resources,_RESOURCE_STATE
from .cfpb_baselines import ACCEPTED,load_rows,build_representation,measured
from .holdout_evaluation import make_baseline,classification_metrics,coefficient_payload
from .text_preprocessing import text_matrix_stats
from .compression import (GRID,SEEDS,ALPHAS,compress,rank_vocabulary,fit_hash_stream,
                          predict_hash_stream,retention,summarize_runs)
BASELINE=ROOT/'validation/cfpb/20261014T204807Z-2270924a'
FULL_ID='primary_linear_svc_balanced_C1'


def run():
    _RESOURCE_STATE.clear();check_resources(before_stage=True)
    now=datetime.now(timezone.utc);rid=now.strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=ROOT/'validation/cfpb'/rid;out.mkdir(exist_ok=False)
    source=ROOT/'data/CCDB_Export_15_January_2026_through_February_2026.csv'
    baseline=json.loads((BASELINE/'manifest.json').read_text());prep=json.loads((ACCEPTED/'manifest.json').read_text())
    protocol=ROOT/'docs/cfpb_compression_protocol.md'
    manifest=dict(run_id=rid,date_utc=now.isoformat(),status='running',python=sys.version,
        root_commit=git_commit(ROOT),reference_commit=baseline['reference_commit'],
        baseline=BASELINE.name,preparation=ACCEPTED.name,baseline_manifest_sha256=sha256_file(BASELINE/'manifest.json'),
        source_sha256=sha256_file(source),protocol_sha256=sha256_file(protocol),
        spec_sha256=sha256_file(ROOT/'docs/cfpb_benchmark_spec.md'),grid=list(GRID),seeds=list(SEEDS),
        classifier='balanced LinearSVC C=1; streaming SGD separate',threads=1,
        code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'src/compression.py']},
        peak_memory_definition='absolute sampled process RSS; not incremental model allocation')
    write_json(out/'manifest.json',manifest)
    runtime=[];raw=[];perclass=[];confusions=[];predictions=[];resources=[];hash_candidates=[];configs=[]
    try:
        if manifest['source_sha256']!=SOURCE_SHA256 or manifest['spec_sha256']!=prep['spec_sha256']:raise ValueError('Source/spec changed')
        for folder,meta in [(ACCEPTED,prep),(BASELINE,baseline)]:
            for name,digest in meta['artifacts_sha256'].items():
                if sha256_file(folder/name)!=digest:raise ValueError(f'Artifact changed: {folder/name}')
        import importlib.metadata
        manifest['packages']={n:importlib.metadata.version(n) for n in baseline['packages']}
        if manifest['packages']!=baseline['packages'] or sys.version!=baseline['python']:raise ValueError('Environment drift')
        write_json(out/'run_plan.json',dict(grid=list(GRID),sketch_seeds=list(SEEDS),hash_alphas=list(ALPHAS),
             streaming_passes=5,batch_size=1024,natural_subset=[1000,5000,50000],protocol_sha256=manifest['protocol_sha256'],
             model_parameters=make_baseline('linear_svc',class_weight='balanced',C=1.).get_params()))
        with measured('rehydrate_offline_curated_text_replay_store',runtime):rows=load_rows(source,ACCEPTED)
        train=rows[rows.primary&rows.split.eq('train')]
        parts={'train':train}
        for split in ['validation','test']:
            parts[split]=rows[rows.primary&rows.split.eq(split)]
            parts['natural_stream_'+split]=rows[rows.natural_stream&rows.split.eq(split)]
        manifest['replay_store_dataframe_bytes']=int(rows.memory_usage(index=True,deep=True).sum())
        manifest['training_label_array_bytes']=train.label.to_numpy().nbytes
        refs=pd.read_csv(BASELINE/'metrics_per_run.csv');refs=refs[refs.configuration_id.eq(FULL_ID)]
        accepted_predictions=pd.read_csv(BASELINE/'predictions.csv',dtype={'complaint_id':str})
        def record(cfg,name,pred,timing,**extra):
            part=parts[name];pop='natural_stream' if name.startswith('natural_stream_') else 'primary';split=name.removeprefix('natural_stream_')
            context=dict(run_id=rid,configuration_id=cfg['id'],method=cfg['method'],k=cfg['k'],seed=cfg['seed'],
                         partition=split,population=pop,model='SGD_log_loss' if cfg['method']=='hashing' else 'balanced_LinearSVC_C1')
            score,classes,cm,norm=classification_metrics(part.label.to_numpy(),pred)
            ref=refs[(refs.population==pop)&(refs.partition==split)].iloc[0]
            row=dict(**context,**score,**cfg['stats'],**coefficient_payload(cfg['model']),
                     fit_time=cfg['fit_time'],transform_time=timing['transform_time'],prediction_time=timing['prediction_time'],
                     training_transform_time=cfg['training_transform_time'],peak_memory=cfg['peak_memory'],
                     evaluation_N=len(part),alpha=cfg.get('alpha'),**extra)
            for metric in ['macro_f1','balanced_accuracy']:
                row.update({metric+'_'+k:v for k,v in retention(score[metric],ref[metric]).items()})
            raw.append(row)
            perclass.extend(dict(**context,**c,product=CLASSES[c['label']]) for c in classes)
            confusions.extend(dict(**context,true_label=i,predicted_label=j,count=int(cm[i,j])) for i in range(4) for j in range(4))
            predictions.extend(dict(**context,complaint_id=cid,true_label=int(y),predicted_label=int(p))
                               for cid,y,p in zip(part.complaint_id,part.label,pred))
            if pop=='natural_stream':
                for stratum in sorted(part.stratum.unique()):
                    mask=part.stratum.eq(stratum).to_numpy();ss,cc,_,_=classification_metrics(part.label.to_numpy()[mask],pred[mask])
                    resources.append(dict(kind='natural_stratum_metric',**context,stratum=stratum,N=int(mask.sum()),**ss))
            return row
        # Hashing is trained before even loading the accepted TF-IDF vocabulary.
        for k in GRID:
            print('Hashing validation',k,flush=True)
            with measured(f'hashing_{k}_five_pass_train_and_selection',runtime) as timing:
                models,fit_times,stats=fit_hash_stream(train.text.tolist(),train.label.to_numpy(),k,check=check_resources)
                candidates=[]
                for alpha,model in models.items():
                    pred,pt=predict_hash_stream(model,parts['validation'].text.tolist(),k,check=check_resources)
                    score,_,_,_=classification_metrics(parts['validation'].label,pred)
                    candidates.append(dict(k=k,alpha=alpha,**score,fit_time=fit_times[alpha],**pt))
            hash_candidates+=candidates
            choice=min(candidates,key=lambda c:(-c['macro_f1'],-c['balanced_accuracy'],-c['alpha']))
            alpha=choice['alpha'];cfg=dict(id=f'hashing_k{k}',method='hashing',k=k,seed=42,alpha=alpha,
               model=models[alpha],stats=stats,fit_time=fit_times[alpha],training_transform_time=stats['hash_seconds_all_passes'],
               peak_memory=timing['peak_rss_bytes'],selection_total_fit_time=sum(fit_times.values()))
            pred,pt=predict_hash_stream(cfg['model'],parts['validation'].text.tolist(),k,check=check_resources)
            record(cfg,'validation',pred,pt);configs.append(cfg)
        pd.DataFrame(hash_candidates).to_csv(out/'hashing_validation_candidates.csv',index=False)
        print('Reconstructing frozen TF-IDF after hashing fits',flush=True)
        parts,matrices,fullstats=build_representation('primary',rows,ACCEPTED,runtime)
        pd.DataFrame(fullstats).to_csv(out/'full_representation_stats.csv',index=False)
        vocabulary=pd.read_csv(ACCEPTED/'vocabulary.csv',keep_default_na=False).token.tolist()
        with measured('January_frequency_ranking',runtime):ranking,totals=rank_vocabulary(train.text.tolist(),vocabulary)
        pd.DataFrame(dict(rank=np.arange(len(ranking)),original_feature_index=ranking,January_term_frequency=totals[ranking])).to_csv(out/'truncation_ranking.csv',index=False)
        def transformed(cfg,name):
            t=time.perf_counter();x=compress(matrices[name],cfg['method'],cfg['k'],seed=cfg['seed'] or 42,ranking=ranking)
            return x,time.perf_counter()-t
        def evaluate(cfg,name):
            if cfg['method']=='hashing':
                with measured(cfg['id']+'_predict_'+name,runtime) as timing:
                    pred,pt=predict_hash_stream(cfg['model'],parts[name].text.tolist(),cfg['k'],check=check_resources)
                cfg['peak_memory']=max(cfg['peak_memory'],timing['peak_rss_bytes'])
                return record(cfg,name,pred,pt)
            with measured(cfg['id']+'_predict_'+name,runtime) as timing:
                x,transform_time=transformed(cfg,name);t=time.perf_counter();pred=cfg['model'].predict(x);predict_time=time.perf_counter()-t
            cfg['peak_memory']=max(cfg['peak_memory'],timing['peak_rss_bytes'])
            st=text_matrix_stats(x)
            resources.append(dict(kind='heldout_representation',configuration_id=cfg['id'],name=name,**{k:v for k,v in st.items() if not k.startswith('qos_')}))
            return record(cfg,name,pred,dict(transform_time=transform_time,prediction_time=predict_time),evaluation_nnz=x.nnz,evaluation_zero_rows=st['zero_rows'])
        # Full endpoint once; reused for all three applicable methods.
        full=None
        for method in ['truncation','bucket','jl']:
            for k in GRID:
                seeds=SEEDS if method in ['bucket','jl'] and k<50000 else [None]
                for seed in seeds:
                    if k==50000 and full is not None:continue
                    cfg=dict(id=f'{method}_k{k}_s{seed}',method=method,k=k,seed=seed)
                    print('SVM validation',cfg['id'],flush=True)
                    with measured(cfg['id']+'_transform_fit',runtime) as timing:
                        x,transform_time=transformed(cfg,'train')
                        model=make_baseline('linear_svc',class_weight='balanced',C=1.)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');t=time.perf_counter();model.fit(x,train.label.to_numpy());fit_time=time.perf_counter()-t
                        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError(f'Nonconvergence: {cfg["id"]}')
                    if not np.isfinite(model.coef_).all():raise ValueError('Nonfinite coefficients')
                    stats=text_matrix_stats(x);stats['csr_bytes_kind']='materialized CSR array payload'
                    cfg.update(model=model,stats=stats,fit_time=fit_time,training_transform_time=transform_time,peak_memory=timing['peak_rss_bytes'])
                    record_row=evaluate(cfg,'validation');configs.append(cfg)
                    if k==50000:
                        full=cfg
                        ref=refs[(refs.population=='primary')&(refs.partition=='validation')].iloc[0]
                        for metric in ['macro_f1','balanced_accuracy','accuracy']:np.testing.assert_allclose(record_row[metric],ref[metric],rtol=0,atol=1e-8)
                        arrays=np.load(BASELINE/(FULL_ID+'_parameters.npz'))
                        np.testing.assert_array_equal(model.coef_,arrays['coef']);np.testing.assert_array_equal(model.intercept_,arrays['intercept'])
        # Freeze all selection decisions, including hash per-k choices, before test.
        provisional=pd.DataFrame(raw)
        aliases=[]
        for method in ['bucket','jl']:
            r=dict(next(r for r in raw if r['configuration_id']==full['id'] and r['partition']=='validation'))
            r.update(method=method,configuration_id=f'{method}_identity_shared',shared_identity_source=full['id']);aliases.append(r)
        vsummary=summarize_runs(pd.DataFrame(raw+aliases))
        minima=[]
        for method,g in vsummary.groupby('method'):
            for cutoff in [.99,.98,.95]:
                good=g[g.macro_f1_retention_mean>=cutoff];minima.append(dict(method=method,cutoff=cutoff,k=int(good.k.min()) if len(good) else None))
        selection=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_predictions_computed=False,
            hash_alpha_by_k={str(c['k']):c['alpha'] for c in configs if c['method']=='hashing'},
            primary_classifier='balanced LinearSVC C=1 unchanged',validation_retention_minima=minima,
            no_test_or_natural_selection=True)
        write_json(out/'validation_selection.json',selection)
        selection_hash=sha256_file(out/'validation_selection.json')
        manifest['test_started_utc']=datetime.now(timezone.utc).isoformat();manifest['validation_selection_sha256']=selection_hash
        write_json(out/'manifest.json',manifest)
        for cfg in configs:
            print('Test',cfg['id'],flush=True);evaluate(cfg,'test')
            if cfg['k'] in [1000,5000,50000]:
                for name in ['natural_stream_validation','natural_stream_test']:evaluate(cfg,name)
            if cfg['k']==50000 and cfg['method']=='truncation':
                for name in ['test','natural_stream_validation','natural_stream_test']:
                    pop='natural_stream' if name.startswith('natural') else 'primary';split=name.removeprefix('natural_stream_')
                    got=pd.DataFrame(predictions);got=got[(got.configuration_id==cfg['id'])&(got.population==pop)&(got.partition==split)].set_index('complaint_id')
                    expected=accepted_predictions[(accepted_predictions.configuration_id==FULL_ID)&(accepted_predictions.population==pop)&(accepted_predictions.partition==split)].set_index('complaint_id')
                    np.testing.assert_array_equal(got.loc[expected.index].predicted_label,expected.predicted_label)
        for method in ['bucket','jl']:
            for row in list(raw):
                if row['configuration_id']==full['id']:
                    r=dict(row);r.update(method=method,configuration_id=f'{method}_identity_shared',shared_identity_source=full['id']);raw.append(r)
        frame=pd.DataFrame(raw);frame.to_csv(out/'compression_raw.csv',index=False)
        summarize_runs(frame).to_csv(out/'compression_summary.csv',index=False)
        pd.DataFrame(perclass).to_csv(out/'per_class_metrics.csv',index=False)
        pd.DataFrame(confusions).to_csv(out/'confusion_matrices.csv',index=False)
        pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
        pd.DataFrame(runtime).to_csv(out/'runtime_memory.csv',index=False)
        pd.DataFrame(resources).to_csv(out/'population_diagnostics.csv',index=False)
        write_json(out/'configuration_resources.json',[{k:v for k,v in c.items() if k!='model'}|coefficient_payload(c['model']) for c in configs])
        if sha256_file(source)!=SOURCE_SHA256 or sha256_file(out/'validation_selection.json')!=selection_hash:raise ValueError('Integrity failure')
        manifest.update(status='completed_pending_validation',swap_monitoring=dict(_RESOURCE_STATE),
           peak_process_rss_bytes=max(t['peak_rss_bytes'] for t in runtime),
           full_baseline_predictions_and_parameters_identical=True,
           artifacts_sha256={p.name:sha256_file(p) for p in out.iterdir() if p.name!='manifest.json'})
        write_json(out/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'manifest.json',manifest);pd.DataFrame(raw).to_csv(out/'partial_results.csv',index=False)
        raise
    return out


def main():
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    with threadpool_limits(limits=1):print(run(),flush=True)

if __name__=='__main__':main()
