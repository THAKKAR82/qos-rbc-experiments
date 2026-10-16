"""Uncompressed CFPB holdout baselines: python -m src.cfpb_baselines.

Consumes immutable accepted curation; never runs grouping or changes Bank code.
All validation choices are written and hashed before any test prediction.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .cfpb import CLASSES, SOURCE_SHA256, clean_narrative, chronological_split
from .cfpb_prepare import (ROOT, sha256_file, write_json, git_commit, check_resources,
                           _RESOURCE_STATE)
from .holdout_evaluation import (make_baseline, classification_metrics, select_validation,
                                 require_sparse, coefficient_payload)
from .text_preprocessing import make_tfidf_vectorizer, make_unigram_vectorizer, text_matrix_stats

ACCEPTED = ROOT / 'validation/cfpb/20261014T192947Z-7b6668f9'


@contextmanager
def measured(stage, runtime):
    """Absolute process RSS sampled every 50ms, not incremental/model memory.

    Memory violations interrupt the main thread at its next Python safe point.
    A native solver may finish a native call before that interrupt is delivered.
    """
    check_resources(before_stage=True)
    stop = threading.Event()
    state = dict(stage=stage, start_rss_bytes=psutil.Process().memory_info().rss,
                 peak_rss_bytes=0, memory_violation=None)
    def sample():
        while not stop.is_set():
            state['peak_rss_bytes'] = max(state['peak_rss_bytes'], psutil.Process().memory_info().rss)
            try:
                check_resources()
            except RuntimeError as exc:
                state['memory_violation'] = str(exc)
                import _thread
                _thread.interrupt_main()
                return
            stop.wait(.05)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        yield state
    finally:
        elapsed = time.perf_counter() - started
        stop.set(); thread.join()
        state['seconds'] = elapsed
        state['peak_rss_bytes'] = max(state['peak_rss_bytes'], psutil.Process().memory_info().rss)
        runtime.append(state)


def configurations():
    """Declared before fitting; only balanced SVM's C is validation-selected."""
    specs=[]
    def add(rep, model, weight=None, **kw):
        identifier=f'{rep}_{model}_{weight or "none"}'
        if 'C' in kw:identifier+=f'_C{kw["C"]:g}'
        if 'alpha' in kw:identifier+=f'_a{kw["alpha"]:g}'
        specs.append(dict(configuration_id=identifier,representation=rep,model=model,
                          class_weight=weight,**kw))
    for model in ['ridge','logistic']:
        for weight in [None,'balanced']:add('primary',model,weight)
    add('primary','linear_svc',C=1.)
    for c in [.1,1.,10.]:add('primary','linear_svc','balanced',C=c)
    for alpha in [10.,1.]:add('unigram','ridge',alpha=alpha)
    for weight in [None,'balanced']:add('primary_float64','ridge',weight)
    return specs


def load_rows(source, preparation):
    """Rehydrate accepted IDs, verify metadata/text hashes, retain no extra rows."""
    rows=pd.read_csv(preparation/'split_manifest.csv',dtype=str,keep_default_na=False)
    if rows.complaint_id.duplicated().any():raise ValueError('Duplicate accepted ID')
    rows['label']=rows.label.astype(int)
    for col in ['primary','natural_stream']:rows[col]=rows[col].eq('True')
    if not rows.split.eq(rows.date_received.map(chronological_split)).all():
        raise ValueError('Partition date mismatch')
    if not rows['product'].eq(rows.label.map(dict(enumerate(CLASSES)))).all():
        raise ValueError('Label order mismatch')
    wanted=set(rows.complaint_id); parts=[]
    for chunk in pd.read_csv(source,usecols=['Complaint ID','Consumer complaint narrative','Product','Date received'],
                             dtype=str,keep_default_na=False,chunksize=100000):
        check_resources();parts.append(chunk[chunk['Complaint ID'].isin(wanted)])
    raw=pd.concat(parts,ignore_index=True)
    rows=rows.merge(raw,left_on='complaint_id',right_on='Complaint ID',validate='one_to_one')
    if set(rows.complaint_id)!=wanted:raise ValueError('Accepted rows missing from source')
    if not (rows['product'].eq(rows.Product)&rows.date_received.eq(rows['Date received'])).all():
        raise ValueError('Source metadata differs from accepted manifest')
    rows['text']=rows['Consumer complaint narrative'].map(clean_narrative)
    accepted={}
    for chunk in pd.read_csv(preparation/'curation_audit.csv',usecols=['complaint_id','model_text_hash'],
                             dtype=str,keep_default_na=False,chunksize=100000):
        accepted.update(chunk[chunk.complaint_id.isin(wanted)].set_index('complaint_id').model_text_hash.to_dict())
    if not all(hashlib.sha256(r.text.encode()).hexdigest()==accepted[r.complaint_id]
               for r in rows.itertuples()):raise ValueError('Model text differs from accepted preparation')
    rows['id_numeric']=rows.complaint_id.astype(int)
    rows=rows.sort_values(['date_received','id_numeric']).reset_index(drop=True)
    primary=rows[rows.primary]
    if primary.template_group_id.duplicated().any():raise ValueError('Primary group overlap')
    for split,n in [('train',13459),('validation',3033),('test',4213)]:
        b=primary[primary.split==split]
        if len(b)!=n or set(b.label)!=set(range(4)):raise ValueError('Frozen population mismatch')
    return rows.drop(columns=['Consumer complaint narrative'])


def build_representation(rep, rows, preparation, runtime):
    """Reproduce January fit exactly; float64 uses frozen primary vocabulary."""
    config='unigram' if rep=='unigram' else 'primary'
    vm=json.loads((preparation/'vectorizer_manifest.json').read_text())[config]
    vocabulary=pd.read_csv(preparation/vm['vocabulary_file'],keep_default_na=False).token.tolist()
    vectorizer=make_unigram_vectorizer() if config=='unigram' else make_tfidf_vectorizer()
    if rep=='primary_float64':
        vectorizer.set_params(dtype=np.float64,vocabulary={word:i for i,word in enumerate(vocabulary)})
    partitions={'train':rows[rows.primary & rows.split.eq('train')]}
    for split in ['validation','test']:
        partitions[split]=rows[rows.primary & rows.split.eq(split)]
        partitions['natural_stream_'+split]=rows[rows.natural_stream & rows.split.eq(split)]
    train=partitions['train']
    if hashlib.sha256('\n'.join(train.complaint_id).encode()).hexdigest()!=vm['training_ids_sha256']:
        raise ValueError('January row order differs from preparation')
    with measured(rep+'_vocabulary_idf_fit',runtime):
        x=vectorizer.fit_transform(train.text.tolist()).tocsr()
    if vectorizer.get_feature_names_out().tolist()!=vocabulary:raise ValueError('Vocabulary drift')
    if rep!='primary_float64':np.testing.assert_array_equal(vectorizer.idf_,np.load(preparation/vm['idf_file']))
    matrices={'train':x}
    for name,part in partitions.items():
        if name=='train':continue
        with measured(rep+'_transform_'+name,runtime):
            matrices[name]=vectorizer.transform(part.text.tolist()).tocsr()
    stats=[]
    expected=pd.read_csv(preparation/'representation_stats.csv')
    for name,matrix in matrices.items():
        matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices();require_sparse(matrix)
        stat=text_matrix_stats(matrix)
        population='natural_stream' if name.startswith('natural_stream_') else 'primary'
        partition=name.removeprefix('natural_stream_')
        if population=='primary' and stat['zero_rows']/stat['N']>.01:raise ValueError('Zero-feature gate')
        original=expected[(expected.configuration_id==config)&(expected.population==population)&
                          (expected.partition==partition)].iloc[0]
        for field in ['N','D','nnz','max_row_nnz','max_column_nnz','zero_rows']:
            if stat[field]!=original[field]:raise ValueError(f'Representation drift: {rep}/{name}/{field}')
        stat.update(representation=rep,population=population,partition=partition)
        if partition!='train':stat['qos_row_only']=stat['qos_row_or_column']=None
        stats.append(stat)
    return partitions, matrices, stats


def run(source, preparation, output_root):
    _RESOURCE_STATE.clear();check_resources(before_stage=True)
    now=datetime.now(timezone.utc);run_id=now.strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output=output_root/run_id;output.mkdir(parents=True,exist_ok=False)
    prep=json.loads((preparation/'manifest.json').read_text())
    protected={str(p):sha256_file(p) for p in preparation.iterdir() if p.is_file()}
    spec=ROOT/'docs/cfpb_benchmark_spec.md';review=ROOT/'docs/cfpb_curation_validity_review.md'
    manifest=dict(run_id=run_id,date_utc=now.isoformat(),status='running',spec_version='1.0',
                  spec_sha256=sha256_file(spec),source_sha256=sha256_file(source),
                  preparation=str(preparation.relative_to(ROOT)),preparation_manifest_sha256=sha256_file(preparation/'manifest.json'),
                  validity_review_sha256=sha256_file(review),
                  review_resolution='Recommendation B; user explicitly authorized uncompressed baselines; secondary evaluation required',
                  frozen_preparation_gates_unchanged=True,root_commit=git_commit(ROOT),
                  reference_commit=prep['reference_commit'],python=sys.version,platform=platform.platform(),
                  packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','psutil']},
                  code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'src/holdout_evaluation.py']},
                  seed=42,threads=1,peak_memory_definition='absolute process RSS sampled at 50ms; includes resident data and other fitted models; not incremental estimator allocation',
                  policy_extensions='User-authorized fixed C=1 unweighted SVM and C=1 SAGA logistic with/without balanced weights; no logistic search',
                  class_weight_rationale='Training counts are imbalanced; macro-F1/balanced accuracy justify a fixed weighted counterpart for each family',
                  classifiers_trained=False)
    runtime=[];metrics=[];perclass=[];confusions=[];predictions=[];fits={};models={};validation=[]
    write_json(output/'manifest.json',manifest)
    try:
        if manifest['source_sha256']!=SOURCE_SHA256 or manifest['spec_sha256']!=prep['spec_sha256']:
            raise ValueError('Frozen source/spec identity mismatch')
        if prep['gates']['blocking'] or not prep['curation_complete']:raise ValueError('Preparation blocked')
        for name,digest in prep['artifacts_sha256'].items():
            if sha256_file(preparation/name)!=digest:raise ValueError(f'Preparation artifact drift: {name}')
        for name,digest in prep['code_sha256'].items():
            if sha256_file(ROOT/name)!=digest:raise ValueError(f'Preparation implementation drift: {name}')
        if manifest['packages']!=prep['packages'] or sys.version!=prep['python']:raise ValueError('Environment drift')
        specs=configurations()
        plan=dict(configurations=[dict(**s,parameters=make_baseline(s['model'],class_weight=s['class_weight'],
                  **{k:s[k] for k in ['alpha','C'] if k in s}).get_params()) for s in specs],
                  selection='Balanced SVM: primary validation macro-F1, balanced accuracy, smaller C. Overall primary winner: same criteria then configuration ID. Fixed controls all retained.',
                  majority='January most frequent class; fixed label order breaks ties',
                  test_policy='No test prediction before validation_selection.json is written and hashed; no refitting after selection')
        write_json(output/'run_plan.json',plan);manifest['run_plan_sha256']=sha256_file(output/'run_plan.json')
        with measured('source_rehydration',runtime):rows=load_rows(source,preparation)
        matrices={};partitions={};stats=[]
        for rep in ['primary','unigram','primary_float64']:
            print('Building',rep,flush=True)
            partitions[rep],matrices[rep],st=build_representation(rep,rows,preparation,runtime);stats+=st
        pd.DataFrame(stats).to_csv(output/'representation_stats.csv',index=False)
        statmap={(s['representation'],s['population'],s['partition']):s for s in stats}
        def evaluate(s, name):
            cid=s['configuration_id'];rep=s['representation'];part=partitions[rep][name];x=matrices[rep][name]
            population='natural_stream' if name.startswith('natural_stream_') else 'primary'
            partition=name.removeprefix('natural_stream_')
            with measured(cid+'_predict_'+name,runtime) as timing:
                pred=(np.full(len(part),majority,dtype=int) if s['model']=='majority' else models[cid].predict(x))
            populations=[(population,np.ones(len(part),dtype=bool))]
            if population=='natural_stream':
                populations += [('natural_stream:'+stratum,part.stratum.eq(stratum).to_numpy())
                                for stratum in sorted(part.stratum.unique())]
            for pop,mask in populations:
                if not mask.any():continue
                y=part.label.to_numpy()[mask];yp=pred[mask]
                score,classes,cm,norm=classification_metrics(y,yp)
                context=dict(configuration_id=cid,representation=rep,model=s['model'],class_weight=s['class_weight'],
                             population=pop,partition=partition,run_id=run_id,spec_version='1.0',seed=42)
                st=statmap[(rep,'primary','train')];fit=fits[cid]
                r=dict(**context,**score,train_N=st['N'],validation_N=len(partitions[rep]['validation']),
                       test_N=len(partitions[rep]['test']),evaluation_N=int(mask.sum()),D=st['D'],nnz=st['nnz'],
                       input_csr_bytes=st['csr_bytes'],input_dtype=st['dtype'],fit_time=fit['fit_time'],
                       prediction_time=timing['seconds'] if pop==population else None,
                       peak_memory=max(fit['peak_memory'],timing['peak_rss_bytes']),
                       prediction_peak_rss_bytes=timing['peak_rss_bytes'],**{k:fit[k] for k in
                         ['coefficient_bytes','intercept_bytes','parameter_bytes','coefficient_dtype','solver','actual_solver']},
                       C=s.get('C',1.),alpha=s.get('alpha',10.) if s['model']=='ridge' else None,
                       notes='January-only fit; no compression. Peak memory is absolute process RSS. Stratum timing is shared with natural-stream prediction; absent-class strata are not four-class comparable.')
                metrics.append(r)
                perclass.extend(dict(**context,**v,product=CLASSES[v['label']]) for v in classes)
                confusions.extend(dict(**context,true_label=i,predicted_label=j,count=int(cm[i,j]),
                                       row_proportion=float(norm[i,j]) if np.isfinite(norm[i,j]) else None)
                                  for i in range(4) for j in range(4))
                if pop==population:
                    predictions.extend(dict(**context,complaint_id=cidrow,true_label=int(yt),predicted_label=int(pt))
                                       for cidrow,yt,pt in zip(part.complaint_id,y,yp))
            return metrics[-len(populations)]
        majority=int(np.bincount(partitions['primary']['train'].label,minlength=4).argmax())
        for s in specs:
            cid=s['configuration_id'];rep=s['representation'];print('Fitting',cid,flush=True)
            model=make_baseline(s['model'],class_weight=s['class_weight'],**{k:s[k] for k in ['alpha','C'] if k in s})
            with measured(cid+'_fit',runtime) as timing, warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(matrices[rep]['train'],partitions[rep]['train'].label.to_numpy())
            messages=[str(w.message) for w in caught]
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                write_json(output/'convergence_failure.json',dict(configuration_id=cid,warnings=messages))
                raise RuntimeError('Nonconverged configuration; stop before test scoring')
            if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():raise ValueError('Nonfinite model')
            models[cid]=model;fits[cid]=dict(fit_time=timing['seconds'],peak_memory=timing['peak_rss_bytes'],
                **coefficient_payload(model),solver=getattr(model,'solver','liblinear'),
                actual_solver=getattr(model,'solver_',None),n_iter=np.asarray(getattr(model,'n_iter_',[])).tolist(),warnings=messages)
            validation.append(evaluate(s,'validation'))
        dummy=dict(configuration_id='primary_majority',representation='primary',model='majority',class_weight=None)
        fits[dummy['configuration_id']]=dict(fit_time=0.,peak_memory=psutil.Process().memory_info().rss,
                 coefficient_bytes=0,intercept_bytes=0,parameter_bytes=0,coefficient_dtype=None,
                 solver='training_majority',actual_solver='training_majority')
        majority_val=evaluate(dummy,'validation')
        candidates=[r for r in validation if r['model']=='linear_svc' and r['class_weight']=='balanced']
        chosen_svm=select_validation(candidates)
        selected=[s for s in specs if not(s['model']=='linear_svc' and s['class_weight']=='balanced') or s['configuration_id']==chosen_svm]
        eligible=[r for r in validation if r['representation']=='primary' and r['configuration_id'] in {s['configuration_id'] for s in selected}]
        strongest=select_validation(eligible)
        selection=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),selected_balanced_svm=chosen_svm,
                       strongest_primary_by_validation=strongest,selected_configurations=selected,
                       validation_candidates=[{k:r[k] for k in ['configuration_id','population','partition','macro_f1','balanced_accuracy','C']} for r in validation],
                       no_test_predictions_yet=not any(r['partition']=='test' for r in metrics),
                       majority_label=majority,majority_validation_macro_f1=majority_val['macro_f1'],
                       learnability_gate_passed=next(r['macro_f1'] for r in eligible if r['configuration_id']==strongest)-majority_val['macro_f1']>=.05)
        write_json(output/'validation_selection.json',selection)
        manifest['validation_selection_sha256']=sha256_file(output/'validation_selection.json')
        manifest['test_started_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'manifest.json',manifest)
        for s in selected+[dummy]:
            for name in ['test','natural_stream_validation','natural_stream_test']:evaluate(s,name)
            if s['model']!='majority':
                model=models[s['configuration_id']]
                np.savez_compressed(output/(s['configuration_id']+'_parameters.npz'),coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
        pd.DataFrame(metrics).to_csv(output/'metrics_per_run.csv',index=False)
        pd.DataFrame(metrics).to_csv(output/'baseline_results.csv',index=False)
        pd.DataFrame(perclass).to_csv(output/'per_class_metrics.csv',index=False)
        pd.DataFrame(confusions).to_csv(output/'confusion_matrices.csv',index=False)
        predictions=pd.DataFrame(predictions);predictions.to_csv(output/'predictions.csv',index=False)
        pd.DataFrame(runtime).to_csv(output/'runtime_memory.csv',index=False);write_json(output/'model_fits.json',fits)
        differences=[]
        for weight in ['none','balanced']:
            for pop in ['primary','natural_stream']:
                for split in ['validation','test']:
                    ids=[f'{rep}_ridge_{weight}' for rep in ['primary','primary_float64']]
                    parts=[predictions[(predictions.configuration_id==i)&(predictions.population==pop)&(predictions.partition==split)].set_index('complaint_id') for i in ids]
                    np.testing.assert_array_equal(parts[0].index,parts[1].index)
                    ms=[next(r for r in metrics if r['configuration_id']==i and r['population']==pop and r['partition']==split) for i in ids]
                    differences.append(dict(class_weight=weight,population=pop,partition=split,
                        prediction_disagreement=float((parts[0].predicted_label!=parts[1].predicted_label).mean()),
                        **{k+'_delta_float64_minus_float32':ms[1][k]-ms[0][k] for k in ['macro_f1','balanced_accuracy','accuracy']}))
        pd.DataFrame(differences).to_csv(output/'dtype_sensitivity.csv',index=False)
        if sha256_file(source)!=manifest['source_sha256']:raise ValueError('Source modified')
        if any(sha256_file(Path(p))!=h for p,h in protected.items()):raise ValueError('Preparation artifacts modified')
        if sha256_file(output/'validation_selection.json')!=manifest['validation_selection_sha256']:raise ValueError('Selection changed after test')
        manifest.update(status='completed_pending_validation',classifiers_trained=True,swap_monitoring=dict(_RESOURCE_STATE),
                        peak_process_rss_bytes=max(r['peak_rss_bytes'] for r in runtime),
                        all_configurations_converged=True,artifacts_sha256={p.name:sha256_file(p) for p in output.iterdir() if p.name!='manifest.json'})
        write_json(output/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',swap_monitoring=dict(_RESOURCE_STATE))
        write_json(output/'manifest.json',manifest);pd.DataFrame(runtime).to_csv(output/'runtime_memory.csv',index=False)
        raise
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/CCDB_Export_15_January_2026_through_February_2026.csv')
    parser.add_argument('--preparation',type=Path,default=ACCEPTED)
    parser.add_argument('--output-root',type=Path,default=ROOT/'validation/cfpb')
    args=parser.parse_args()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    with threadpool_limits(limits=1):print(run(args.source,args.preparation,args.output_root),flush=True)

if __name__=='__main__':main()
