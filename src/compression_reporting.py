"""Presentation-only tables and plots from frozen compression outputs."""
import sys,json,os
from pathlib import Path
import numpy as np,pandas as pd


def write_outputs(run):
    os.environ['MPLCONFIGDIR']='/tmp/qos-compression-mpl'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=Path(run);raw=pd.read_csv(r/'compression_raw.csv');summary=pd.read_csv(r/'compression_summary.csv')
    primary=summary[summary.population=='primary'];labels={'truncation':'Vocabulary truncation + SVM','bucket':'Balanced bucketing + SVM','jl':'Sparse JL + SVM','hashing':'Hashing + SGD'}
    colors=dict(zip(labels,['#1b6ca8','#d88318','#258343','#a24da5']))
    comparison=primary.pivot(index=['method','k'],columns='partition',values=['macro_f1_mean','balanced_accuracy_mean','accuracy_mean','macro_f1_retention_mean','balanced_accuracy_retention_mean'])
    comparison.columns=['_'.join(c) for c in comparison.columns];comparison.reset_index().to_csv(r/'validation_test_comparison.csv',index=False)
    minima=[];best=[]
    for split,g in primary.groupby('partition'):
     for method,b in g.groupby('method'):
      for cut in [.99,.98,.95]:
       q=b[b.macro_f1_retention_mean>=cut];minima.append(dict(partition=split,method=method,macro_f1_retention_threshold=cut,minimum_grid_k=int(q.k.min()) if len(q) else None,interpretation='selection eligible' if split=='validation' else 'descriptive only; not selection'))
     for k,b in g.groupby('k'):
      f=b.sort_values(['macro_f1_mean','balanced_accuracy_mean','method'],ascending=[False,False,True]).iloc[0]
      a=b.sort_values(['balanced_accuracy_mean','macro_f1_mean','method'],ascending=[False,False,True]).iloc[0]
      best.append(dict(partition=split,k=k,macro_f1_best_method='/'.join(sorted(b.loc[b.macro_f1_mean.eq(f.macro_f1_mean),'method'])),macro_f1=f.macro_f1_mean,
                       balanced_accuracy_best_method='/'.join(sorted(b.loc[b.balanced_accuracy_mean.eq(a.balanced_accuracy_mean),'method'])),balanced_accuracy=a.balanced_accuracy_mean,
                       interpretation='validation comparison' if split=='validation' else 'descriptive; no reselection'))
    pd.DataFrame(minima).to_csv(r/'retention_thresholds.csv',index=False);pd.DataFrame(best).to_csv(r/'best_method_by_dimension.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,split in enumerate(['validation','test']):
     for row,metric in enumerate(['macro_f1','balanced_accuracy']):
      ax=axes[row,col]
      for method in labels:
       g=primary[(primary.method==method)&(primary.partition==split)].sort_values('k')
       ax.plot(g.k,g[metric+'_mean'],marker='o',label=labels[method],color=colors[method])
       sd=g[metric+'_std'].fillna(0);ax.fill_between(g.k,g[metric+'_mean']-sd,g[metric+'_mean']+sd,color=colors[method],alpha=.12)
      ax.set(xscale='log',xlabel='Dimension k',ylabel=metric.replace('_',' '),title=split.title());ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('CFPB primary population: seed means ± sample SD (singletons have no band)');fig.tight_layout();fig.savefig(r/'performance_vs_dimension.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in labels:
     g=primary[(primary.method==method)&(primary.partition=='test')].sort_values('k')
     axes[0].plot(g.csr_bytes_mean/2**20,g.macro_f1_mean,'o-',label=labels[method],color=colors[method])
     axes[1].plot(g.parameter_bytes_mean/2**20,g.macro_f1_mean,'o-',label=labels[method],color=colors[method])
    axes[0].set(xlabel='Training CSR MiB (hashing: calculated, not materialized)',ylabel='Test macro-F1')
    axes[1].set(xlabel='Coefficient + intercept MiB',ylabel='Test macro-F1')
    for ax in axes:ax.grid(alpha=.25)
    axes[1].legend(fontsize=8);fig.suptitle('Classical representation/model payload; not whole-process RSS');fig.tight_layout();fig.savefig(r/'performance_vs_classical_resources.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for method in labels:
     g=primary[(primary.method==method)&(primary.partition=='test')].sort_values('k')
     axes[0].plot(g.k,g.fit_time_mean,'o-',label=labels[method],color=colors[method])
     axes[1].plot(g.k,g.training_transform_time_mean,'o-',color=colors[method])
     axes[2].plot(g.k,g.peak_memory_mean/2**20,'o-',color=colors[method])
    for ax,title,y in zip(axes,['Fit only','Training transform (hashing: five passes)','Absolute sampled process RSS'],['seconds','seconds','MiB']):
     ax.set(xscale='log',xlabel='Dimension k',ylabel=y,title=title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Offline transforms exclude shared TF-IDF construction; RSS includes resident data and prior results');fig.tight_layout();fig.savefig(r/'runtime_memory.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in labels:
     g=primary[(primary.method==method)&(primary.partition=='test')].sort_values('k')
     for ax,col in zip(axes,['qos_row_only_mean','qos_row_or_column_mean']):ax.plot(g[col],g.macro_f1_mean,'o-',color=colors[method],label=labels[method])
    for ax,title in zip(axes,['Bank row-only convention','Reference row-or-column convention']):
     ax.set(xlabel='Theoretical LS-SVM accounting units (per binary subproblem)',ylabel='Test macro-F1',title=title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Conditional theoretical proxy only: not bytes, hardware qubits, runtime or multiclass cost');fig.tight_layout();fig.savefig(r/'theoretical_qos_accounting.png',dpi=180);plt.close(fig)
    raw[raw.population=='primary'][['method','k','seed','partition','nnz','csr_bytes','parameter_bytes','max_row_nnz','max_column_nnz','qos_row_only','qos_row_or_column']].query("partition == 'validation'").to_csv(r/'training_resource_accounting.csv',index=False)
    pd.DataFrame([dict(method=m,k=50000,source_method='truncation',source_configuration='truncation_k50000_sNone',n_runs=1) for m in ['bucket','jl']]).to_csv(r/'identity_aliases.csv',index=False)
    print(pd.DataFrame(best).to_string(index=False));print(pd.DataFrame(minima).to_string(index=False))


if __name__ == "__main__":
    write_outputs(sys.argv[1])
