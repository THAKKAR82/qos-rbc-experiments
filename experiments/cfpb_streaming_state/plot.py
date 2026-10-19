"""Standalone pilot figures from scalar validation summaries; no fitting."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(output):
    output=Path(output)
    frame=pd.read_csv(output/'trajectory_summary.csv')
    frame=frame[frame.status.eq('completed')]
    refs=json.loads((output/'preparation.json').read_text())['references']
    colors=dict(hash_sgd='#2371a4',sparse_jl_sgd='#df831d',awm='#2a916b',mission='#965aa6')
    for metric in ('accuracy','balanced_accuracy'):
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
        for row,task in enumerate(('credit_vs_checking','mortgage_vs_rest')):
            for col,rep in enumerate(('A_frozen','B_causal')):
                ax=axes[row,col]
                subset=frame[frame.task.eq(task)&frame.representation.eq(rep)]
                for method,color in colors.items():
                    for order,style in [('chronological','-'),('random','--')]:
                        points=subset[subset.method.eq(method)&subset.ordering.eq(order)].sort_values('peak_model_state_bytes')
                        ax.plot(points.peak_model_state_bytes/1024,100*points['primary_'+metric],
                                linestyle=style,marker='o',color=color,label=method+' / '+order)
                ax.axhline(100*refs[task]['selected']['primary'][metric],color='black',lw=1,label='Full Ridge')
                ax.set_xscale('log',base=2);ax.grid(alpha=.2)
                ax.set_title(task.replace('_',' ')+' · '+rep)
                ax.set_xlabel('Allocated algorithm state (KiB)')
                ax.set_ylabel('Validation '+metric.replace('_',' ')+' (%)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('Untuned one-pass pilot · seed 42 · lines join tested points only')
        fig.savefig(output/(metric+'_vs_state.png'),dpi=180)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');plot(p.parse_args().output)
