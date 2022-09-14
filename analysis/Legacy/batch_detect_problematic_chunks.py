# Imperial College London. 2022

# Version history/updates:
# 2022 Apr - Sina Hafezi - initial version including almost all metrics except SIR, HASPI & HASQI
# 2022 Jun - Sina Hafezi - added museval for SIR


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sounddevice as sd
from pystoi import stoi
from scipy.signal import resample
from pesq import pesq
import sys
import MBSTOI
import haspi
import speechmetrics as sm
import pysepm
import museval

refPath = '/Volumes/HD FRL/data/SPEAR/Extra/Train/'
procPath = '/Volumes/HD FRL/data/SPEAR Enhanced/Passthrough/'
savePath = ''; # where output (metric scores csv) will be saved

method_name = 'passthrough'
seg_file = 'orig_segments_D1_S1.csv' # segments/chunks database

metrics = ['MBSTOI','STOI','ESTOI',
           'PESQ','PESQ-NB',
           'SegSNR','fwSegSNR',
           'SI-SDR','SDR','ISR','SAR','HASPI']

metrics2 = ['MBSTOI','STOI','ESTOI','SegSNR','fwSegSNR','HASPI','SM']


# Setting up columns for metric matrix
isMBSTOI = 'MBSTOI' in metrics
if isMBSTOI: metrics.remove('MBSTOI')
side_str = ['L','R']
# 'cols' are the name of columns in metric matrix
cols = ['%s (%s)' % (x,y) for x in metrics for y in side_str] # creating 2x (Left & Right) mono-based metric
if isMBSTOI: cols.insert(0,'MBSTOI') # stereo-based metric

# Read in chunks info
segments = pd.read_csv(seg_file)
nSeg=len(segments)
#nSeg=279-21
#nSeg=60

isProblem = np.zeros((nSeg,len(metrics2)))

# Loop through chunks
metric_vals=[]
for n in range(nSeg):
    print('Segment: ' + str(n+1) + '/' + str(nSeg))
    seg = segments.loc[n]
    dataset = int(seg['dataset'][1:])
    session = seg['session']
    file_name = seg['file_name']
    minute_name = file_name[0:2]
    target_ID = seg['target_ID']
    sample_start = seg['sample_start']-1
    sample_stop = seg['sample_stop']-1
    
    # refenrece signal
    ref_file = '%sDataset_%d/%s/Session_%d/%s/ref_D%d_S%d_M%s_ID%d.wav' % (refPath,dataset,'Reference_Audio',session,minute_name,dataset,session,minute_name,target_ID)
    x_ref, fs_ref = sf.read(ref_file,start=sample_start,stop=sample_stop)
    if len(x_ref.shape)<2: x_ref = np.tile(x_ref,(2,1)).transpose() # mono to stereo conversion if needed
    
    # processed signal
    proc_file = '%sDataset_%d/Session_%d/D%d_S%d_M%s_ID_%d.wav' % (procPath,dataset,session,dataset,session,minute_name,target_ID)
    x_proc, fs_proc = sf.read(proc_file,start=sample_start,stop=sample_stop)
    if len(x_proc.shape)<2: x_proc = np.tile(x_proc,(2,1)).transpose() # mono to stereo conversion if needed
        
    fs = fs_ref 
    
    try:
        # Speech Metric setup
        window = x_ref.shape[0] / fs # sec, whole segment (for speechmetrics)
        spmetrics = sm.load('relative',window)
        SM_scores = []
        SM_scores.append(spmetrics(x_proc[:,0], x_ref[:,0], rate=fs)) # Left
        SM_scores.append(spmetrics(x_proc[:,1], x_ref[:,1], rate=fs)) # Right
    except:
        isProblem[n,metrics2.index('SM')]=1

    # Loop through columns/metrics (some are already calculated/some to be calculated)
    scores=[]
    for m in cols:
        try: 
            if m=='MBSTOI': # stereo-based metric (get it once)
                score = MBSTOI.mbstoi(x_ref[:,0], x_ref[:,1], x_proc[:,0], x_proc[:,1], fs)
                mm = m
                pp = mm # pacakge name. one of metrics2
            else: # mono-based metrics (get it for each channel)
                xx = m.split(' (')  # format: 'metric (channel)'
                mm = xx[0] # metric 
                pp = mm # pacakge name. one of metrics2
                cc = int('R' in xx[1]) # channel index (L: 0 /R: 1)
                if mm=='STOI': 
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=False)
                elif mm=='ESTOI':
                    score = stoi(x_ref[:,cc], x_proc[:,cc], fs, extended=True)
                elif mm=='SDR':
                    #score = float(SM_scores[cc]['sdr'][0])
                    pp = 'SM'
                elif mm=='ISR':
                    #score = float(SM_scores[cc]['isr'][0])
                    pp = 'SM'
                elif mm=='SAR':
                    #score = float(SM_scores[cc]['sar'][0])
                    pp = 'SM'
                elif mm=='SI-SDR':
                    #score = float(SM_scores[cc]['sisdr'][0])
                    pp = 'SM'
                elif mm=='PESQ':
                    #score = float(SM_scores[cc]['pesq'][0]) 
                    pp = 'SM'
                elif mm=='PESQ-NB':
                    #score = float(SM_scores[cc]['nb_pesq'][0])  
                    pp = 'SM'
                elif mm=='SegSNR':
                    score = pysepm.SNRseg(x_ref[:,cc], x_proc[:,cc], fs)
                elif mm=='fwSegSNR':
                    score = pysepm.fwSNRseg(x_ref[:,cc], x_proc[:,cc], fs)   
                elif mm=='HASPI':
                    score, _ = haspi.haspi_v2(x_ref[:,cc], fs, x_proc[:,cc], fs, [0,0,0,0,0,0])
        except:
            isProblem[n,metrics2.index(pp)]=1
        else:
            isProblem[n,metrics2.index(pp)]=int(np.isnan(score))
                
 
        #scores.append(score)       
    #metric_vals.append(scores)

validities = pd.DataFrame(isProblem,columns=metrics2)
validities.to_csv(savePath + 'D1_S1_chunks_validity_' + method_name + '.csv', index=False)