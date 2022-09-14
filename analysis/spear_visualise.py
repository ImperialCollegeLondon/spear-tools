# Imperial College London. 2022

# Version history/updates:
# 2022 Jun - Sina Hafezi - initial version, boxplot of absolute and relative metrics (vs passthrough)
# 2022 Jul - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Jul - Sina Hafezi - nan removal bug fix 
# 2022 Jul - Sina Hafezi - making the paths concatenation robust to operating sys using os.path
# 2022 Aug - Alastair Moore - update to bash control
# 2022 Sep - Pierre Guiraud - final debugging and finishing touches

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from pathlib import PurePath


def spear_visualise(output_dir, metrics_ref, metrics_proc, reference_name='passthrough', method_name='baseline'):

    # ensure the output directory is present before we start running anything
    # - to allow running supplementary results files which already exist will not
    #   be replaced
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Visualisation settings
    matplotlib.rcParams.update({'font.size': 18})

    # Read in chunks info
    seg_pass = pd.read_csv(metrics_ref)
    seg_pass = seg_pass.drop(['global_index', 'file_name', 'chunk_index'], axis=1)

    seg_proc = pd.read_csv(metrics_proc)
    seg_proc = seg_proc.drop(['global_index', 'file_name', 'chunk_index'], axis=1)

    nSeg=len(seg_pass)


    gg=.2 # separation between L/R boxplots
    sides = ['L','R'] 
    cols = list(seg_pass.keys()) # columns of csv (metrics)
    nCol = len(cols)
    visited_metrics = []
    # Loop through columns
    for n in range(nCol):
        col_name = cols[n]
        metric_name = col_name.split(' (')[0]
        if not metric_name in visited_metrics: 
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
            fig.suptitle(metric_name)
            fig.set_figwidth(8)
            
            if '(' in col_name:
                # monoral metric
                # find its pair
                side_name = col_name.split(' (')[1].split(')')[0] # L or R
                oside_name = sides[int(not bool(sides.index(side_name)))] # opposite side 
                n2 = cols.index('%s (%s)' % (metric_name,oside_name)) # index of opposite side
                
                s_pass = np.array([ seg_pass[cols[n]] , seg_pass[cols[n2]] ]).T  # [nChunk x 2]
                s_proc = np.array([ seg_proc[cols[n]] , seg_proc[cols[n2]] ]).T


                # Remove NaN values
                nan_ids = np.where(np.isnan(s_pass[:,0]) | np.isnan(s_pass[:,1]) | np.isnan(s_proc[:,0]) | np.isnan(s_proc[:,1]))
                s_pass = np.array([np.delete(s_pass[:,0],nan_ids),np.delete(s_pass[:,1],nan_ids)]).T
                s_proc = np.array([np.delete(s_proc[:,0],nan_ids),np.delete(s_proc[:,1],nan_ids)]).T
                
                if side_name=='R': # making sure the order of side is L then R
                    s_pass = np.flip(s_pass,axis=1)
                    s_proc = np.flip(s_proc,axis=1)

                
                
                # Absolute score (Processed & Reference)
                ax = axs[0]
                ax.boxplot(np.array([s_pass[:,0],s_pass[:,1],s_proc[:,0],s_proc[:,1]]).T,positions = [1-gg,1+gg,3-gg,3+gg])
                ax.set_xticks([1,3])
                ax.xaxis.set_ticklabels([reference_name,method_name])
                ax.yaxis.grid(True)
                ax.set_ylabel('%s'%(metric_name))
                # Relative score (Processed - Reference)
                ax = axs[1]
                ax.boxplot(s_proc-s_pass,positions = [1-gg,1+gg])
                ax.yaxis.grid(True)
                ax.set_ylabel('\u0394%s'%(metric_name))
                ax.set_xticks([1])
                ax.xaxis.set_ticklabels([method_name])
                z_line = ax.plot(ax.get_xlim(),np.zeros((2,)),'--k')
                z_line[0].set_label('Ref.')
                ax.legend()
            else:
                # binaural metric (e.g. MBSTOI)
                s_pass = np.array(seg_pass[cols[n]])
                s_proc = np.array(seg_proc[cols[n]])

                # Remove NaN values
                nan_ids = np.where(np.isnan(s_pass) | np.isnan(s_proc))
                s_pass = np.delete(s_pass,nan_ids)
                s_proc = np.delete(s_proc,nan_ids)
                
                # Absolute score (Processed & Reference)
                ax = axs[0]
                ax.boxplot(np.array([s_pass,s_proc]).T)
                ax.yaxis.grid(True)
                ax.set_ylabel('%s'%(metric_name))
                ax.xaxis.set_ticklabels([reference_name,method_name])
                # Relative score (Processed - Reference)
                ax = axs[1]
                ax.boxplot(s_proc-s_pass)
                ax.yaxis.grid(True)
                ax.set_ylabel('\u0394%s'%(metric_name))
                ax.xaxis.set_ticklabels([method_name])
                z_line = ax.plot(ax.get_xlim(),np.zeros((2,)),'--k')
                z_line[0].set_label('Ref.')
                ax.legend()
            
            fig.tight_layout()
            
            # Save figures
            fig.savefig(os.path.join(output_dir,f'{metric_name}.png'),bbox_inches='tight')
            
            visited_metrics.append(metric_name)


if __name__ == '__main__':
    # parse the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root",
                        help="Directory below which to save metrics plots")                    
    parser.add_argument("metrics_ref",
                        help="Metrics file (.csv) for the reference method")
    parser.add_argument("metrics_proc",
                        help="Metrics file (.csv) for the current method")
    parser.add_argument("-r","--reference_name",
                        help="reference algorithm used: [passthrough] by default",
                        default='passthrough')  
    parser.add_argument("-m","--method_name",
                        help="enhancement algorithm used: [baseline] by default",
                        default='baseline')                   
    args = parser.parse_args()
    print(args)

    spear_visualise(args.output_root, args.metrics_ref, args.metrics_proc, reference_name=args.reference_name, method_name=args.method_name)

        