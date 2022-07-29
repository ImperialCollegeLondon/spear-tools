""" 
Function to load the correct path based on the current session (from 1 to 15), desired set current option

ATTENTION: You need to have run the SPEAR directory creation file and updated dir_cwd to create your own SPEAR

Valid options are:
Misc (do not use dataset_n or session_n)
    "noise"
    "hoa"
    "ATF"
Main:
    "array"
    "DOA"
    "orientation"
Extra
    "reference"
    "Tascar"
    "VAD"
    "PosOri"

"""

import os
from pathlib import Path
from pathlib import PurePath


def PathFinder(dataset_n, session_n, option):

    dir_cwd = str(Path(os.path.realpath(__file__)).parents[1]) + '/SPEAR/'


    dir_dataset = f'Dataset_{dataset_n}'
    dir_session = f'Session_{session_n}'

    if option in ['array', 'DOA', 'orientation']:
        dir_main = 'Main'

        if option == 'array':
            dir_active = 'Microphone_Array_Audio'
        elif option == 'DOA':
            dir_active = 'DOA_sources'
        elif option == 'orientation':
            dir_active = 'Array_Orientation'

    elif option in ['reference', 'Tascar', 'VAD', 'PosOri']:
        dir_main = 'Extra'

        if option == 'reference':
            dir_active = 'Reference_Audio'
        elif option == 'Tascar':
            dir_active = 'TASCAR'
        elif option == 'VAD':
            dir_active = 'VAD'
        elif option == 'PosOri':
            dir_active = 'Reference_PosOri'


    if session_n < 10:
        dir_release = 'InitialRelease'
        dir_set = 'Train'
    elif session_n < 13:
        dir_release = 'InitialRelease'
        dir_set  = 'Dev'
    elif session_n < 16:
        dir_release = 'FinalRelease'
        dir_set  = 'Eval'
   

    if option in ['noise', 'hoa', 'ATF']:
        dir_main = 'Miscellaneous'
        if option=='noise':
            dir_active = 'AmbientNoise'
        elif option=='hoa':
            dir_active = 'HOA_weights'
        elif option=='ATF':
            dir_active = 'Array_Transfer_Functions'

        Path_full = PurePath(dir_cwd, dir_main, dir_active)

    else:
        Path_full = PurePath(dir_cwd, dir_release, dir_main, dir_set, dir_dataset, dir_active, dir_session)

    return Path_full