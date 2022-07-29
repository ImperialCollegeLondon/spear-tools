# Imperial College London. 2022

# Version history/updates:
# 2022 Jun - Sina Hafezi - initial version, boxplot of absolute and relative metrics (vs passthrough)
# 2022 Jul - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Jul - Sina Hafezi - nan removal bug fix 
# 2022 Jul - Sina Hafezi - making the paths concatenation robust to operating sys using os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
from pathlib import PurePath




def BatchVisualisation(save_path, proc_file, pass_file, ToSave=0, passthrough_only=False):
    if (not os.path.exists(save_path)) and bool(ToSave):
        os.makedirs(save_path)


    # Visualisation settings
    matplotlib.rcParams.update({'font.size': 18})

    # Read in chunks info
    seg_pass = pd.read_csv(pass_file)
    if not passthrough_only:
        seg_proc = pd.read_csv(proc_file)
    else:
        seg_proc = np.zeros(np.shape(seg_pass))
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
                if not passthrough_only:
                    s_proc = np.array([ seg_proc[cols[n]] , seg_proc[cols[n2]] ]).T
                else:
                    s_proc = np.array(np.zeros(np.shape(s_pass)))

                # Remove NaN values
                nan_ids = np.where(np.isnan(s_pass[:,0]) | np.isnan(s_pass[:,1]) | np.isnan(s_proc[:,0]) | np.isnan(s_proc[:,1]))
                s_pass = np.array([np.delete(s_pass[:,0],nan_ids),np.delete(s_pass[:,1],nan_ids)]).T
                s_proc = np.array([np.delete(s_proc[:,0],nan_ids),np.delete(s_proc[:,1],nan_ids)]).T
                
                if side_name=='R': # making sure the order of side is L then R
                    s_pass = np.flip(s_pass,axis=1)
                    s_proc = np.flip(s_proc,axis=1)

                
                
                # Absolute score (Processed & Passthrough)
                ax = axs[0]
                ax.boxplot(np.array([s_pass[:,0],s_pass[:,1],s_proc[:,0],s_proc[:,1]]).T,positions = [1-gg,1+gg,3-gg,3+gg])
                ax.set_xticks([1,3])
                ax.xaxis.set_ticklabels(['Passthrough','Processed'])
                ax.yaxis.grid(True)
                ax.set_ylabel('%s'%(metric_name))
                # Relative score (Processed - Passthrough)
                ax = axs[1]
                ax.boxplot(s_proc-s_pass,positions = [1-gg,1+gg])
                ax.yaxis.grid(True)
                ax.set_ylabel('\u0394%s'%(metric_name))
                ax.set_xticks([1])
                ax.xaxis.set_ticklabels(['Processed'])
                z_line = ax.plot(ax.get_xlim(),np.zeros((2,)),'--k')
                z_line[0].set_label('Pass.')
                ax.legend()
            else:
                # binaural metric (e.g. MBSTOI)
                s_pass = np.array(seg_pass[cols[n]])
                if not passthrough_only:
                    s_proc = np.array(seg_proc[cols[n]])
                else:
                    s_proc = np.array(np.zeros(np.shape(s_pass)))

                # Remove NaN values
                nan_ids = np.where(np.isnan(s_pass) | np.isnan(s_proc))
                s_pass = np.delete(s_pass,nan_ids)
                s_proc = np.delete(s_proc,nan_ids)
                
                # Absolute score (Processed & Passthrough)
                ax = axs[0]
                ax.boxplot(np.array([s_pass,s_proc]).T)
                ax.yaxis.grid(True)
                ax.set_ylabel('%s'%(metric_name))
                ax.xaxis.set_ticklabels(['Passthrough','Processed'])
                # Relative score (Processed - Passthrough)
                ax = axs[1]
                ax.boxplot(s_proc-s_pass)
                ax.yaxis.grid(True)
                ax.set_ylabel('\u0394%s'%(metric_name))
                ax.xaxis.set_ticklabels(['Processed'])
                z_line = ax.plot(ax.get_xlim(),np.zeros((2,)),'--k')
                z_line[0].set_label('Pass.')
                ax.legend()
            
            fig.tight_layout()
            
            if ToSave: fig.savefig(os.path.join(save_path,f'{metric_name}.png'),bbox_inches='tight')
            
            visited_metrics.append(metric_name)


if __name__ == '__main__':
    path_SSD = str(Path(os.path.realpath(__file__)).parents[1])
    
    ToSave = 1 # (optional) 0: dont/ 1: do save the plots

    list_cases = ['D2',1, 'M00']

    metric_name    = 'baseline'
    metric_name_pt = 'passthrough'
    case_str       = ''
    for case in list_cases:
        if type(case)==str:
            case_str    += case
        else:
            case_str    += 'S'+str(case)

    save_path = str(PurePath(path_SSD, 'SPEAR_ProcessedAudio', case_str)) # path where plots will be saved

    proc_file = os.path.join(save_path,'metrics_' + metric_name    + case_str + '.csv')
    pass_file = os.path.join(save_path,'metrics_' + metric_name_pt + case_str + '.csv')  # provided metrics csv file for passthrough

    BatchVisualisation(save_path, proc_file, pass_file, ToSave=ToSave)
        