# Imperial College London. 2022

# Version history/updates:
# 2022 Jun - Sina Hafezi
# 2022 July - Pierre Guiraud - updated into a function to be used in the batch_master script
# 2022 Aug - Alastair Moore - update to bash control
# 2022 Sep - Pierre Guiraud - final debugging and finishing touches

import argparse
import os
from pathlib import Path
from pathlib import PurePath
import numpy as np
import soundfile as sf
from SPEAR import *
from Processor import *

def spear_enhance(input_root, output_dir, list_cases=[], method_name='baseline'):
    """
    input_root: folder to find SPEAR data
    output_root: folder to save processed audio
    """
    fs = int(48e3) # output freq
    mics = [] # mics subset
    out_chan = [5,6]

    method_name_allowed_values = ['passthrough','baseline']
    if method_name not in method_name_allowed_values:
        raise ValueError(f'method_name must be one of {method_name_allowed_values}')

    # ensure the output directory is present before we start running anything
    # - to allow running supplementary results files which already exist will not
    #   be replaced
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print('Setting up Dataset...')
    sp = SPEAR_Data(input_root)

    if method_name == 'baseline':
        print('Setting up Processor...')
        proc = SPEAR_Processor(sp.get_all_AIRs(),fs=fs,mics=mics,out_chan=out_chan)

    cases=sp.get_cases(list_cases)
    print(cases)
    nCase = len(cases)
    print('Batch Processing...')
    for n in range(nCase):
        this_case = cases.loc[n]
        dataset = this_case['dataset']
        session = this_case['session']
        file = 'D%d_S%d_M%s' % (dataset,session,this_case['file'])
        print('\n--+ %d/%d (%%%2.2f) %s +--' % (n+1,nCase,100*n/nCase,file))
        
        sp.set_file(dataset,session,file)
        targets = sp.get_targets()
        nTarget = len(targets)


        if method_name == 'passthrough':
            # load in only the reference channels, with resampling if required
            # passthrough doesn't do any enhancement so is independent of
            # the target
            y, _  = sp.get_array_audio(fs=fs, mics=out_chan)
            
            for ti in range(nTarget):
                target = targets[ti]
                output_file = Path(output_dir, f'{file}_ID{target}.wav')
                print(output_file)
                sf.write(output_file, y.T, fs)
                
        elif method_name == 'baseline':
            # array_audio, array_fs = sp.get_array_audio()
            # TODO: create a method to expose array rotation contained in sp.ht
            # e.g.
            # array_rotation = sp.get_array_rotation()

            for ti in range(nTarget):
                print('Target: %d/%d'%(ti+1,nTarget))
                target = targets[ti]                
                output_file = Path(output_dir, f'{file}_ID{target}.wav')
                
                # avoid re processing already existing files
                if output_file.is_file():
                    print(f'{output_file} already processed')
                    continue
                
                # pass in timestamped target directions     
                target_doas_as_unit_vectors = sp.get_pos_samples(target)
                proc.set_target_pos(target_doas_as_unit_vectors)
                
                # do the processing
                _, _ = proc.process_file(sp.array_file,out_file=output_file)
        else:
            print(f'method_name {method_name} was not handled')
 
    print('\nProcessing Finished!')








if __name__ == '__main__':
    # parse the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root",
                        help="Directory containing Dataset_n folder")
    parser.add_argument("output_root",
                        help="Directory below which to save processed audio")                    
    parser.add_argument("-l", "--list_cases",
                        help="list cases",
                        nargs="+", type=str,
                        default=[])
    parser.add_argument("-m","--method_name",
                        help="enhancement algorithm to use: [baseline] or passthrough",
                        default='baseline')                   
    args = parser.parse_args()
    print(args)
    
    list_cases = args.list_cases
    if len(list_cases) > 1:
        list_cases[1] = int(list_cases[1])
    if len(list_cases) > 3:
        raise ValueError('list cases must have a maximum of 3 items')

    spear_enhance(args.input_root, args.output_root, list_cases=list_cases, method_name=args.method_name)