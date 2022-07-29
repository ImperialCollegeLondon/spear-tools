# Imperial College London. 2022

# Version history/updates:
# 2022 Jun - Sina Hafezi - initial script
# 2022 Jul - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Jul - Sina Hafezi - making the paths concatenation robust to operating sys using os.path

import os
from pathlib import Path
from pathlib import PurePath
import numpy as np
import soundfile as sf
from SPEAR import *
from Processor import *

def BatchProcessing(root_path, save_path, list_cases=[], method_name='baseline', passthrough=False):

    fs = int(48e3) # output freq
    mics = [] # mics subset
    out_chan = [5,6]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('Setting up Dataset...')
    sp = SPEAR_Data(root_path)

    if not passthrough:
        print('Setting up Processor...')
        proc = SPEAR_Processor(sp.get_all_AIRs(),fs=fs,mics=mics,out_chan=out_chan)

    cases=sp.get_cases(list_cases)
    nCase = len(cases)
    print('Batch Processing...')
    for n in range(nCase):
        this_case = cases.loc[n]
        dataset = this_case['dataset']
        session = this_case['session']
        file = 'D%d_S%d_%s' % (dataset,session,this_case['file'])
        print('\n--+ %d/%d (%%%2.2f) %s +--' % (n+1,nCase,100*n/nCase,file))
        
        sp.set_file(dataset,session,file)
        targets = sp.get_targets()
        nTarget = len(targets)

        if passthrough:
            y, _  = sp.get_array_audio()
            y = y[np.array(out_chan)-1,:]

        for ti in range(nTarget):
            print('Target: %d/%d'%(ti+1,nTarget))
            target = targets[ti]
            
            if not passthrough:
                proc.set_target_pos(sp.get_pos_samples(target))
                saveName = os.path.join(save_path,method_name,f'Dataset_{dataset}',f'Session_{session}')
            else:
                saveName = os.path.join(save_path,'Passthrough',f'Dataset_{dataset}',f'Session_{session}')

            if not os.path.exists(saveName):
                os.makedirs(saveName)
                
            saveName = os.path.join(saveName,'%s_ID_%d.wav' % (file,target))

            ### Adding a loop to avoid re processing already existing files
            if os.path.exists(saveName):
                print(f'{saveName} already processed')
                continue
            
            if not passthrough:
                _, _ = proc.process_file(sp.array_file,out_file=saveName)
            else:
                sf.write(saveName, y.T, fs)
 
    print('\nProcessing Finished!')

if __name__ == '__main__':
    path_SSD = str(Path(os.path.realpath(__file__)).parents[1])

    root_path = str(PurePath(path_SSD, 'SPEAR', 'InitialRelease', 'Main', 'Train')) # root path for dataset
    save_path = str(PurePath(path_SSD, 'SPEAR_ProcessedAudio'))  # where output (enhanced/processed audio files) will be saved
    list_cases = ['D2',1,'M00']

    BatchProcessing(root_path, save_path, list_cases=list_cases, passthrough=False)