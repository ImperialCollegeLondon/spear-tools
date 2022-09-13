# Imperial College London. 2022

# Version history/updates:
# 2022 May - Sina Hafezi - initial version including all metrics: MBSTOI, STOI, ESTOI, PESQ, PESQ-NB, SegSNR, fwSegSNR, SI-SDR, SDR, ISR, SAR, HASPI
# 2022 Jul - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Jul - Sina Hafezi - making the paths concatenation robust to operating sys using os.path
# 2022 Aug - Pierre Guiraud - modify the output csv to get reference columns

import os
from pathlib import Path
from pathlib import PurePath
import pandas as pd
import numpy as np
import soundfile as sf
from pystoi import stoi
import MBSTOI
import haspi
import speechmetrics as sm
import pysepm


def BatchMetrics(x_proc, x_ref, fs_ref, cols):
    fs = fs_ref 
    # Speech Metric setup
    window = x_ref.shape[0] / fs # sec, whole segment (for speechmetrics)
    spmetrics = sm.load('relative',window)
    SM_scores = []
    try:
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
        try:
            if m=='MBSTOI': # stereo-based metric (get it once)
                score = MBSTOI.mbstoi(x_ref[:,0], x_ref[:,1], x_proc[:,0], x_proc[:,1], fs)
            else: # mono-based metrics (get it for each channel)
                xx = m.split(' (')  # format: 'metric (channel)'
                mm = xx[0] # metric 
                cc = int('R' in xx[1]) # channel index (L: 0 /R: 1)

                if mm=='STOI': 
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=False)
                elif mm=='ESTOI':
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=True)
                elif mm=='SDR':
                    score = float(SM_scores[cc]['sdr'][0])
                elif mm=='ISR':
                    score = float(SM_scores[cc]['isr'][0])
                elif mm=='SAR':
                    score = float(SM_scores[cc]['sar'][0])
                elif mm=='SI-SDR':
                    score = float(SM_scores[cc]['sisdr'][0])
                elif mm=='PESQ':
                    score = float(SM_scores[cc]['pesq'][0]) 
                elif mm=='PESQ-NB':
                    score = float(SM_scores[cc]['nb_pesq'][0])  
                elif mm=='SegSNR':
                    score = pysepm.SNRseg(x_ref[:,cc], x_proc[:,cc], fs)
                elif mm=='fwSegSNR':
                    score = pysepm.fwSNRseg(x_ref[:,cc], x_proc[:,cc], fs)   
                elif mm=='HASPI':
                    score, _ = haspi.haspi_v2(x_ref[:,cc], fs, x_proc[:,cc], fs, [0,0,0,0,0,0])
        except:
            score = np.nan

        scores.append(score)       
    return scores



def BatchEvaluation(root_path, save_path, proc_path, list_cases=[], method_name='baseline', passthrough_only=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the name of the smaller passthrough case
    passthr_name = 'passthrough'

    metrics = ['MBSTOI','STOI','ESTOI',
            'PESQ','PESQ-NB',
            'SegSNR','fwSegSNR',
            'SI-SDR','SDR','ISR','SAR','HASPI']

    # Setting up columns for metric matrix
    isMBSTOI = 'MBSTOI' in metrics
    if isMBSTOI: metrics.remove('MBSTOI')
    side_str = ['L','R']
    # 'cols' are the name of columns in metric matrix
    cols = ['%s (%s)' % (x,y) for x in metrics for y in side_str] # creating 2x (Left & Right) mono-based metric
    if isMBSTOI: cols.insert(0,'MBSTOI') # stereo-based metric
    cols_csv = ['global_index', 'file_name', 'chunk_index'] + cols

    # Read in chunks info
    current_set = Path(root_path).parts[-1]
    seg_file = f'segments_{current_set}.csv' # segments/chunks database
    segments = pd.read_csv(seg_file)

    if len(list_cases)>0:
        segments = segments[segments['dataset']==list_cases[0]]
        dataset_n = int(list_cases[0][1])
        datasets  = range(dataset_n,dataset_n+1)
        method_name_bis  = method_name + '_'+list_cases[0]
        passthr_name += '_'+list_cases[0]
    else:
        datasets = range(1,5)

    for dataset in datasets:
        # Get the passtrough results of the entire target set (if it exists) and get them through the same selection of cases
        metrics_ref = f'metrics_passthrough_D{dataset}.csv'
        passthrough_extract = False
        if os.path.exists(metrics_ref):
            seg_passthr = pd.read_csv(metrics_ref)
            passthrough_extract = True

        # Keep only the desired cases
        if len(list_cases)>1:
            if passthrough_extract: seg_passthr = seg_passthr[segments['session']==list_cases[1]]
            segments      = segments[segments['session']==list_cases[1]]
            method_name_bis  += 'S'+ str(list_cases[1])
            passthr_name += 'S'+ str(list_cases[1])
            if len(list_cases)>2:
                if passthrough_extract: seg_passthr = seg_passthr[[minutes[-3:]==list_cases[2] for minutes in segments['file_name']]]
                segments      = segments[[minutes[-3:]==list_cases[2] for minutes in segments['file_name']]]
                method_name_bis  += list_cases[2]
                passthr_name += list_cases[2]

        nSeg=len(segments)
        
        # Loop through chunks
        metric_vals=[]
        metric_vals_passthrough=[]
        for n in range(nSeg): #616
            print('Segment: ' + str(n+1) + '/' + str(nSeg))
            seg = segments.iloc[n]
            session = seg['session']
            file_name = seg['file_name']
            minute_name = file_name[-2:]
            target_ID = seg['target_ID']
            sample_start = seg['sample_start']-1
            sample_stop = seg['sample_stop']-1
            
            # reference signal
            root_path_ref = str(PurePath(root_path, f'Dataset_{dataset}', 'Reference_Audio', f'Session_{session}')).replace('Main', 'Extra')
            ref_file = os.path.join(root_path_ref,minute_name,'ref_D%d_S%d_M%s_ID%d.wav' % (dataset,session,minute_name,target_ID))
            x_ref, fs_ref = sf.read(ref_file,start=sample_start,stop=sample_stop+1)
            if len(x_ref.shape)<2: x_ref = np.tile(x_ref,(2,1)).transpose() # mono to stereo conversion if needed

            # get chunk info
            chunk_info = [seg['global_index'], file_name, seg['chunk_index']]
            
            # processed signal
            if not passthrough_only:
                proc_file = os.path.join(proc_path,method_name,f'Dataset_{dataset}',f'Session_{session}','D%d_S%d_M%s_ID_%d.wav' % (dataset,session,minute_name,target_ID))
                x_proc, fs_proc = sf.read(proc_file,start=sample_start,stop=sample_stop+1)
                if len(x_proc.shape)<2: x_proc = np.tile(x_proc,(2,1)).transpose() # mono to stereo conversion if needed

                scores = BatchMetrics(x_proc, x_ref, fs_ref, cols)
                metric_vals.append(chunk_info + scores)

            if not passthrough_extract:
                # passtrough signals
                proc_file = os.path.join(proc_path,'Passthrough',f'Dataset_{dataset}',f'Session_{session}','D%d_S%d_M%s_ID_%d.wav' % (dataset,session,minute_name,target_ID))
                x_proc, fs_proc = sf.read(proc_file,start=sample_start,stop=sample_stop+1)
                if len(x_proc.shape)<2: x_proc = np.tile(x_proc,(2,1)).transpose() # mono to stereo conversion if needed

                scores = BatchMetrics(x_proc, x_ref, fs_ref, cols)
                metric_vals_passthrough.append(chunk_info + scores)   
            

        if not passthrough_only:
            metric_vals = pd.DataFrame(metric_vals,columns=cols_csv)
            metric_name = 'metrics_' + method_name_bis + '.csv'
            metric_vals.to_csv(os.path.join(save_path,metric_name), index=False)

        # Save smaller passthrough extract
        metric_name_pt = 'metrics_' + passthr_name + '.csv'
        if passthrough_extract:
            seg_passthr.to_csv(os.path.join(save_path,metric_name_pt), index=False)
        else:
            metric_vals_passthrough = pd.DataFrame(metric_vals_passthrough,columns=cols_csv)
            metric_vals_passthrough.to_csv(os.path.join(save_path,metric_name_pt), index=False)



if __name__ == '__main__':
    path_SSD = str(Path(os.path.realpath(__file__)).parents[1])

    save_path = str(PurePath(path_SSD, 'SPEAR_Analysis')) # where output (metric scores csv) will be saved
    proc_path = str(PurePath(path_SSD, 'SPEAR_ProcessedAudio')) # root path for enhanced/processed audio files
    
    list_cases = ['D2',1, 'M00']

    BatchEvaluation(path_SSD, save_path, proc_path, list_cases=list_cases)