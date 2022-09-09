"""

Master file to run the baseline processing, metrics calculation and visualisation, all at once.
All scripts can be run independently.

PARAMETERS
You can run the analysis on a subset of the dataset using the following notation:
1) all datasets (default)                 -> e.g. list_cases = [] 
2) one dataset + all sessions             -> e.g. list_cases = ['D1']
3) one dataset + one session + all files  -> e.g. list_cases = ['D1',3]
4) one dataset + one session + one file   -> e.g. list_cases = ['D1',3,'M01']

PATHS
The SPEAR dataset folder is assumed to be in the same directory as the tools folder. If not modify root_path.
The path directory for the output processed audio and metrics is selected in proc_path.

"""

import os
from pathlib import Path
from pathlib import PurePath
from batch_Processing import BatchProcessing
from batch_Evaluation import BatchEvaluation
from batch_Visualisation import BatchVisualisation


### Parameters
choose_set  = 'Dev' # Choose to work in Train/Dev/Eval set
list_cases  = ['D3'] # Choose subset to investigate. Use [] if all the sets are of interest.
ToSave      = 1 # 0: dont/ 1: do save the plots
method_name = 'baseline' # Name of the currently tested method. Used for the output csv file name and processed audio folder
passthrough_only = False # Compute only the passthrough metrics and not the processed audio (baseline by default).

### Choose which section to run
run_processing    = False # Output processed and passthrough audio
run_evaluation    = True # Output csv file with all computed metrics on all chunks
run_visualisation = False # Output plots for all metrics

########################

### Paths
if len(list_cases)>1:
    if list_cases[1]<10:
        choose_set = 'Train'
    elif list_cases[1]>12:
        choose_set = 'Eval'
    else:
        choose_set = 'Dev'
# path_SSD = str(Path(os.path.realpath(__file__)).parents[1]) # select the directory one step above the current one
path_SSD = '/Volumes/HD FRL/' # select the directory one step above the current one
root_path = str(PurePath(path_SSD, 'CoreDev_1.0', 'Main', choose_set)) # root path for dataset
proc_path = str(PurePath(path_SSD, 'CoreDev_1.0 Processed')) # where outputs (enhanced/processed audio files, metrics and plots) will be saved


########################


### Part 1: Processing
if run_processing:
    if not passthrough_only:
        print('Running processed files')
        BatchProcessing(root_path, proc_path, list_cases=list_cases, method_name=method_name, passthrough=False)
    print('Running passtrough files')
    BatchProcessing(root_path, proc_path, list_cases=list_cases, method_name=method_name, passthrough=True)


### Part 2: Evaluation
method_name_pt = 'passthrough'
case_str       = ''
for case in list_cases:
    if type(case)==str:
        case_str    += case
    else:
        case_str    += 'S'+str(case)

metricNplot_path = os.path.join(proc_path,case_str) # path where plots will be saved

if run_evaluation:
    BatchEvaluation(root_path, metricNplot_path, proc_path, list_cases=list_cases, method_name=method_name, passthrough_only=passthrough_only)


### Part 3: Visualisation
proc_file = os.path.join(metricNplot_path,'metrics_' + method_name    + '_' + case_str + '.csv')
pass_file = os.path.join(metricNplot_path,'metrics_' + method_name_pt + '_' + case_str + '.csv')  # provided metrics csv file for passthrough

if run_visualisation:
    BatchVisualisation(metricNplot_path, proc_file, pass_file, ToSave=ToSave, passthrough_only=passthrough_only)