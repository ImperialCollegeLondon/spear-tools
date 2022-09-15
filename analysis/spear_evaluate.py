# Imperial College London. 2022

# Version history/updates:
# 2022 May - Sina Hafezi - initial version including all metrics: MBSTOI, STOI, ESTOI, PESQ, PESQ-NB, SegSNR, fwSegSNR, SI-SDR, SDR, ISR, SAR, HASPI
# 2022 Jul - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Jul - Sina Hafezi - making the paths concatenation robust to operating sys using os.path
# 2022 Aug - Alastair Moore - update to bash control
# 2022 Sep - Pierre Guiraud - final debugging and finishing touches

import argparse
import os
from pathlib import Path
from pathlib import PurePath
import pandas as pd
import numpy as np
import soundfile as sf
from pystoi import stoi
# import MBSTOI
from clarity.evaluator import mbstoi
# import haspi
from clarity.evaluator import haspi
import speechmetrics as sm
import pysepm
import sys

def compute_metrics(x_proc, x_ref, fs_ref, cols):
    fs = fs_ref 
    
    
    spmetrics_names = ['SDR','ISR','SAR','SI-SDR','PESQ','PESQ-NB']
    if any(metric_name.split(' (')[0] in spmetrics_names for metric_name in cols):
        # TODO: choose specific metrics computed rather than always doing them all
        
        # Speech Metric setup
        window = x_ref.shape[0] / fs # sec, whole segment (for speechmetrics)
        SM_scores = []
        try:
            spmetrics = sm.load('relative',window)
            SM_scores.append(spmetrics(x_proc[:,0], x_ref[:,0], rate=fs)) # Left
            print('compute_metrics: Left done')        
            SM_scores.append(spmetrics(x_proc[:,1], x_ref[:,1], rate=fs)) # Right
            print('compute_metrics: Right done')                
            SM_error = False
        except Exception as err:
            print('Error in spmetrics')
            print(type(err))
            print(err)
            SM_error = True
        

    # Loop through columns/metrics (some are already calculated/some to be calculated)
    scores=[]
    for m in cols:
        print(f'{m}')
        try:
            if m=='MBSTOI': # stereo-based metric (get it once)
                mm = m
                score = mbstoi.mbstoi(x_ref[:,0], x_ref[:,1], x_proc[:,0], x_proc[:,1], fs)
            else: # mono-based metrics (get it for each channel)
                xx = m.split(' (')  # format: 'metric (channel)'
                mm = xx[0] # metric 
                cc = int('R' in xx[1]) # channel index (L: 0 /R: 1)
                if mm=='STOI': 
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=False)
                elif mm=='ESTOI':
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=True)
                elif mm=='SDR':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['sdr'][0])
                elif mm=='ISR':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['isr'][0])
                elif mm=='SAR':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['sar'][0])
                elif mm=='SI-SDR':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['sisdr'][0])
                elif mm=='PESQ':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['pesq'][0]) 
                elif mm=='PESQ-NB':
                    if SM_error: raise ValueError
                    score = float(SM_scores[cc]['nb_pesq'][0])  
                elif mm=='SegSNR':
                    score = pysepm.SNRseg(x_ref[:,cc], x_proc[:,cc], fs)
                elif mm=='fwSegSNR':
                    score = pysepm.fwSNRseg(x_ref[:,cc], x_proc[:,cc], fs)   
                elif mm=='HASPI':
                    score, _ = haspi.haspi_v2(x_ref[:,cc], fs, x_proc[:,cc], fs, [0,0,0,0,0,0])
                else:
                    raise ValueError(f'Unknown metric: {mm}')
        except:
            print(f'Error while obtaining {mm} metric - assigning NaN')
            score = np.nan

        scores.append(score)       
    return scores



def spear_evaluate(spear_root, proc_dir, segments_file, save_path,
                   list_cases=[], metrics=''):
                              

    # use location in file hierarchy to determine whether it's Train/Dev/Eval
    spear_root = Path(spear_root)
    choose_set = spear_root.name
    choose_set_allowed_values = ['Train','Dev','Eval']
    if choose_set not in choose_set_allowed_values:
        print(f'Expected {choose_set} to be one of {choose_set_allowed_values}')
        sys.exit()   
    if spear_root.parent.name!='Main':
        print(f'Expected {spear_root} to be one directory below "Main"')
        sys.exit()
    ref_root = Path(spear_root,'..','..','Extra',choose_set)

    # don't want to overwrite data so do some checks
    save_path = Path(save_path)
    # - extension should be csv
    if save_path.suffix!='.csv':
        print(f'save_path should specify a .csv file')                     
    # - ensure the parent directory is present before we start running anything
    save_path.parent.mkdir(parents=True, exist_ok=True)    
    # - ensure the output file doesn't already exist
    if save_path.is_file():
        print(f'File already exists at {save_path}. Please choose a different filename or move the old one to avoid loss of data.')
        sys.exit()

    # defines the time periods for each file where it is valid to compute metrics
    segments = pd.read_csv(segments_file)
    if len(list_cases)>0:
        segments = segments[segments['dataset']==list_cases[0]]
        if len(list_cases)>1:
            segments = segments[segments['session']==list_cases[1]]
            if len(list_cases)>2:
                segments = segments[segments['minute']==int(list_cases[2])]

    

    # choice of metrics to run
    available_metrics = ['MBSTOI','STOI','ESTOI',
                         'PESQ','PESQ-NB',
                         'SegSNR','fwSegSNR',
                         'SI-SDR','SDR','ISR','SAR','HASPI']

    if metrics is None:
        metrics = available_metrics
    else:
        if any(item not in available_metrics for item in metrics):
            print(f'Metric must be one of {available_metrics}')
            sys.exit()

    # Setting up columns for metric matrix
    isMBSTOI = 'MBSTOI' in metrics
    if isMBSTOI: metrics.remove('MBSTOI')
    side_str = ['L','R']
    # 'cols' are the name of columns in metric matrix
    cols = ['%s (%s)' % (x,y) for x in metrics for y in side_str] # creating 2x (Left & Right) mono-based metric
    if isMBSTOI: cols.insert(0,'MBSTOI') # stereo-based metric


    cols_csv = ['global_index', 'file_name', 'chunk_index'] + cols

    # Loop through chunks
    metric_vals=[]
    nSeg=len(segments)
    header_was_written = False
    ended_early = False
    for n in range(nSeg):
        print('Segment: ' + str(n+1) + '/' + str(nSeg))
        seg = segments.iloc[n]
        dataset = int(seg['dataset'][1]) #intseg['dataset'][1]) # integer
        session = seg['session'] # integer
        minute = seg['minute'] # integer
        file_name = seg['file_name'] # was original EasyCom name e.g. 01-00-288, now vad_, no nothing
        target_ID = seg['target_ID'] # integer
        sample_start = seg['sample_start']-1
        sample_stop = seg['sample_stop']-1
        
        # get chunk info
        chunk_info = [seg['global_index'], file_name, seg['chunk_index']]
    
        # filenames are subtely different!
        proc_file_name = 'D%d_S%d_M%02d_ID%d.wav' % (dataset,session,minute,target_ID)
        ref_file_name = 'ref_D%d_S%d_M%02d_ID%d.wav' % (dataset,session,minute,target_ID)
        
        # processed signal
        proc_file = Path(proc_dir, proc_file_name)
        print(proc_file)
        
        # allow for processing only a subset of files
        if not proc_file.is_file():
            # missing file skipped
            print('File not found. Skipping...')
        else:

            # reference signal
            ref_file = Path(ref_root, f'Dataset_{dataset}', 'Reference_Audio',
                            f'Session_{session}', f'{minute:02d}', ref_file_name)
            if not ref_file.is_file():
                print(f'Expected reference file at {ref_file} is missing. Attempt to save before aborting...')
                ended_early = True
                break
        
            # debugging
            print(f'{proc_file_name}: sample_start: {sample_start}, sample_stop: {sample_stop}')
            
            
            
            # read in the audio
            x_proc, fs_proc = sf.read(proc_file, start=sample_start, stop=sample_stop+1)
            if len(x_proc.shape)<2: x_proc = np.tile(x_proc, (2,1)).transpose() # mono to stereo conversion if needed
            
            x_ref, fs_ref = sf.read(ref_file, start=sample_start, stop=sample_stop+1)
            if len(x_ref.shape)<2: x_ref = np.tile(x_ref,( 2,1)).transpose() # mono to stereo conversion if needed
 
            # actually compute the metrics on this chunk
            scores = compute_metrics(x_proc, x_ref, fs_ref, cols)
            # metric_vals.append(chunk_info + scores)
            # metric_vals = pd.DataFrame(metric_vals, columns=cols_csv)            
            metric_vals_df = pd.DataFrame([chunk_info + scores], columns=cols_csv)
            
            if not header_was_written:
                metric_vals_df.to_csv(save_path, index=False)
                header_was_written = True
            else: 
                metric_vals_df.to_csv(save_path,
                 index=False,
                 header=False,
                 mode='a')
    
    if ended_early:
        sys.exit()
        




if __name__ == '__main__':    
    # parse the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root",
                        help="Directory containing Dataset_n folder")
    parser.add_argument("proc_dir",
                        help="Directory below which to find processed audio")                    
    parser.add_argument("segments_file",
                        help="csv file containg list of chunks where metrics are valid")
    parser.add_argument("save_path",
                        help="csv file where results should be stored")
    parser.add_argument("-l", "--list_cases",
                        help="list cases",
                        nargs="+", type=str,
                        default=[])
    parser.add_argument("-m","--metrics",
                        help="list a subset of metrics to compute (default is to compute them all)",
                        default=None, nargs='+')                   
    args = parser.parse_args()
    print(args)

    list_cases = args.list_cases
    if len(list_cases) > 1:
        list_cases[1] = int(list_cases[1])
    if len(list_cases) > 3:
        raise ValueError('list cases must have a maximum of 3 items')
    
    metrics = args.metrics

    spear_evaluate(args.input_root, args.proc_dir, args.segments_file, args.save_path,
                       list_cases=list_cases, metrics=metrics)  