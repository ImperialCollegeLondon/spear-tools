# Imperial College London. 2022
# Dataset class for SPEAR Challenge

# Version history: date/author/updates:
# 2022 May 06 - Sina Hafezi - initial version including all read-in/plotting/processing utilities of audio data & metadata
# 2022 Jul 25 - Sina Hafezi - replacing ht/pos interpolant function with ht/pos sample values (while still enabling interpolation, main interpolation moved to Processor class)
# 2022 Jul 28 - Sina Hafezi - making the paths concatenation robust to operating sys using os.path

import glob
import h5py
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

class SPEAR_Data:
    fps = 20 # OptiTrack rate
    file_duration = 60 # sec
    total_frames = file_duration * fps # total number of OptiTrack frames per file
    total_sources = 7 # total number of possible participants
    wearer_ID = 2 # participant ID wearing the array

        
    root_path = '' # main root path to dataset - where 'Dataset_X' folders are
    # subfolders
    ht_folder = 'Array_Orientation'
    array_folder = 'Microphone_Array_Audio'
    doa_folder = 'DOA_sources'
    vad_folder = 'VAD'
    # data & metadata file names
    atf_file = os.path.join('Miscellaneous','Array_Transfer_Functions','Device_ATFs.h5')
    ht_file = '' # to be initialised
    vad_file = '' # to be initialised
    array_file = '' # to be initialised
    # prefix for varying subfolders
    dataset_prefix = 'Dataset_'
    session_prefix = 'Session_'
    # initial case (dataset/session/file)
    dataset = 1
    session = 1
    file = '' # to be initialized
    dataset_folder = '' # to be initialised
    session_folder = '' # to be initialised
    
    ht = [] # (ndarray) head-tracking quaternions interpolant function  (t,qx,qy,qz,qw) as [total_frames x 5]
    t = [] # (ndarray) OptiTrack time frames in sec [total_frames x 1]
    IDs = [] # list of participant IDs present in the file
    src_pos = [] # (dict of cartesian position, keys: source ID) cartesian position relative to the (rotated) array [total_frames x 4] (t,x,y,z) ndarray
    
    
    def __init__(obj,inpath):
        # DESCRIPTION: initialises the class based on SPEAR dataset 
        # *** INPUTS ***
        # inpath    (str) root path to SPEAR dataset        
        obj.root_path = inpath
        obj.update_folders()
        
    def update_folders(obj):
        # DESCRIPTION: set the subfolders to the case's dataset and session
        obj.dataset_folder = '%s%d' % (obj.dataset_prefix,obj.dataset)
        obj.session_folder = '%s%d' % (obj.session_prefix,obj.session)
        
    def set_file(obj,dataset,session,filename):
        # DESCRIPTION: set the folders/files for case of interest and update the affected metadata
        obj.dataset = dataset
        obj.session = session
        obj.file = filename
        obj.update_folders()
        
        
        obj.ht_file = os.path.join(obj.root_path,obj.dataset_folder,obj.ht_folder,obj.session_folder,f'ori_{obj.file}.csv')
        obj.array_file = os.path.join(obj.root_path,obj.dataset_folder,obj.array_folder,obj.session_folder,f'array_{obj.file}.wav')
        obj.vad_file = os.path.join(obj.root_path,'..','..','Extra',obj.root_path.split(os.sep)[-1],obj.dataset_folder,obj.vad_folder,obj.session_folder,f'vad_{obj.file}.csv')
        obj.load_ht()
        obj.set_participant_IDs()
        obj.load_pos()
        
    def get_all_AIRs(obj):
        # DESCRIPTION: returns dictionary of array's Acoustic Impulse Responses (AIRs) for all measured directions
        # *** OUTPUTS ***
        # AIR        (dict) dictionary {'IR': (nSample,nDirection,nChannel),'fs': (int),'directions': (N,2),'nChan': (int)}
        AIR = {'IR': [],'fs': [],'directions': [],'nChan': [], 'azi': [], 'ele': []}
        # IR: (ndarray) Impulse Responses [nSample x nDirection x nChan]
        # fs: (int) sample rate in Hz
        # directions: (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
        # nChan: (int) number of array's sensor/channel
        # azi: sorted unique azimuths (radians) [nDirection x 1]
        # ele: sorted unique elevations (radians) [nDirection x 1]
        f = h5py.File(os.path.join(obj.root_path,'..','..',obj.atf_file),'r')
        #groups = list(f.keys())
        AIR['fs'] = int(list(f['SamplingFreq_Hz'])[0][0])
        AIR['IR'] = np.array(f['IR']) # (ndarray) [nSample x nDirection x nChan]
        AIR['ele'] = (np.pi/2)-np.array(f['Theta']) # (ndarray) elevation in radians [1 x nDirection]
        AIR['azi'] = np.array(f['Phi']) # (ndarray) azimuth in radians [1 x nDirection]
        AIR['directions'] = np.concatenate((AIR['azi'],AIR['ele']),axis=0).T # (ndarray) [nDirection x 2]
        AIR['ele'] = np.sort(np.unique(AIR['ele'])) # (ndarray) [nElevation x 1]
        AIR['azi'] = np.sort(np.unique(AIR['azi'])) # (ndarray) [nAzimuth x 1]
        AIR['nChan'] = AIR['IR'].shape[-1]
        f.close()
        return AIR
    
    def get_targets(obj):
        # DESCRIPTION: returns the participant ID of targets (all sources except ego-centric wearer)
        # *** OUTPUTS ***
        # targets  (list) ID of all potential targets wrt. wearer
        targets = list(np.array(obj.IDs)[np.where(np.array(obj.IDs)!=obj.wearer_ID)[0]])
        return targets
    
    def cart2sph(obj,x,y,z):
        # DESCRIPTION: converts cartesian to spherical coordinate
        # *** INPUTS ***
        # x  (ndarray) x-coordinate(s) [N x 1]
        # y  (ndarray) y-coordinate(s) [N x 1]
        # z  (ndarray) z-coordinate(s) [N x 1]
        # *** OUTPUTS ***
        # az  (ndarray) azimuth(s) in radians [N x 1]
        # el  (ndarray) elevation(s) in radians [N x 1]
        # r   (ndarray) range(s) in radians [N x 1]
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    def sph2cart(obj,az,el,r):
        # DESCRIPTION: converts spherical to cartesian coordinate
        # *** INPUTS ***
        # az  (ndarray) azimuth(s) in radians [N x 1]
        # el  (ndarray) elevation(s) in radians [N x 1]
        # r   (ndarray) range(s) in radians [N x 1]
        # *** OUTPUTS ***
        # x  (ndarray) x-coordinate(s) [N x 1]
        # y  (ndarray) y-coordinate(s) [N x 1]
        # z  (ndarray) z-coordinate(s) [N x 1]
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
    
    def load_ht(obj):
        # DESCRIPTION: load the head-rotation of the wearer and initialize holder param
        data = pd.read_csv(obj.ht_file)  #(nFrame,6) DataFrame [index time qx qy qz qw]
        data = data.to_numpy()
        obj.t = data[:,1] #(nFrame,1)
        obj.ht = data[:,1:] # (nFrame,5) [t qx qy qz qw]    
        
    def set_participant_IDs(obj):
        # DESCRIPTION: find and set the ID of participants present in a file
        path = os.path.join(obj.root_path,obj.dataset_folder,obj.doa_folder,obj.session_folder) 
        files = glob.glob(os.path.join(path,f'doa_{obj.file}_ID*.csv'))
        files.sort()
        obj.IDs = []
        for f in files:
            obj.IDs.append(int(f[-5]))
        
    def load_pos(obj):
        # DESCRIPTION: load cartesian position of present sources and initialize holder param (dict)
        obj.src_pos = dict.fromkeys(np.array(obj.IDs))
        file = os.path.join(obj.root_path,obj.dataset_folder,obj.doa_folder,obj.session_folder,f'doa_{obj.file}_ID')
        for si in range(0,len(obj.IDs)):
            data = pd.read_csv(file + str(obj.IDs[si]) + '.csv')
            data = data.to_numpy()  # (nFrame,4) [index time azi el] in degress
            t = data[:,1]
            az = np.deg2rad(data[:,2])
            el = np.deg2rad(data[:,3])
            x, y, z = obj.sph2cart(az,el, np.ones(az.shape))
            obj.src_pos[obj.IDs[si]] = np.array([t,x,y,z]).T #(nFrame,4) # (dict of cartesian positions) [t x y z] wrt. to wearer
    
    
    def get_doa(obj,src_ID,t):
        # DESCRIPTION: return the doas of a requested participant ID over requested times
        # *** INPUTS ***
        # src_ID  (int) participant ID of interest
        # t     (ndarray) time vector of interest in seconds [N x 1]
        # *** OUTPUTS *** 
        # doas  (ndarray) vector of DOAs (azimuth,elevation) in degrees [N x 2]
        pos_interp = interp1d(obj.src_pos[src_ID][:,0],obj.src_pos[src_ID][:,1:],kind='previous',axis=0,fill_value='extrapolate')
        pos = pos_interp(t)
        doas = np.zeros((len(t),2))
        az, el, r = obj.cart2sph(pos[:,0],pos[:,1],pos[:,2]) 
        doas = np.array([az,el]).T
        doas = np.rad2deg(doas)
        return doas
    
    def get_pos_samples(obj,src_ID):
        # DESCRIPTION: returns the cartesian position samples of reqeusted participant ID
        # *** INPUTS ***
        # src_ID  (int) participant ID of interest
        # *** OUTPUTS *** 
        # (ndarray) relative cartesian positions samples over time [t x y z] (nFrame,4)
        return obj.src_pos[src_ID]
    
    def rotate_sys(obj,q,point):
        # DESCRIPTION: returns cartesian of a point after rotating the cartesian system
        # *** INPUTS ***
        # q       (ndarray)  rotation quartenions (xyzw) of the system [1 x 4]
        # point   (ndarray) cartesian position (xyz) of a point [1 x 3] in default world coordinate system
        # *** OUTPUTS ***
        # point   (ndarray) cartesian position (xyz) of a point [1 x 3] in rotated coordinate system
        Rot = Rotation.from_quat(q)
        point = np.dot(point,Rot.apply(np.eye(3)).transpose())
        return point
    
    def get_array_audio(obj,fs=None,start_t=0.0,duration=None,mics=[]):
        # DESCRIPTION: returns the multi-channel array audio signals
        # *** INPUTS ***
        # fs       (int) optional - output sample rate in Hz, default: original sample rate
        # start_t   (float) optional - start time of segment in sec, default: beginning of the file
        # duration   (float) optional - duration of segment in sec, default: entirely to the end of the file
        # mics      (list) optional - mics/channels subset (indexing from 1), default: all channels
        # *** OUTPUTS ***
        # y   (ndarray) output array audio signal [nSample x nChannel]
        # fs  (int)     output sample rate in Hz
        y, fs = librosa.load(obj.array_file,sr=fs,mono=False,offset=start_t,duration=duration)
        if mics:
            y = y[np.array(mics)-1,:]
        return y, fs
    
    def get_cases(obj,option=[]):
        # DESCRIPTION: returns data frame table of cases (dataset,session,file) of interest
        # Possible level of access (via option):
        # 1) all datasets (default)                 -> e.g. option = [] 
        # 2) one dataset + all sessions             -> e.g. option = ['D1']
        # 3) one dataset + one session + all files  -> e.g. option = ['D1',3]
        # 4) one dataset + one session + one file   -> e.g. option = ['D1',3,'M01']
        # *** INPUTS ***
        # option   (list) [] or ['Dataset'] or ['Dataset',Session] or ['Dataset,Session,'File']
        # *** OUTPUTS ***
        # cases     (Pandas DataFrame) row: a case, columns: 'dataset','session','file'
        
        nDatasets = 4
        c = {'dataset': [], 'session': [], 'file': []}
        datasets = []
        sessions = []
        files = []
        if not option:
            datasets = list(np.arange(nDatasets)+1)
        else:
            datasets.append(int(option[0][1:]))
            if len(option)>1:
                sessions.append(option[1])
                if len(option)>2:
                    files.append(option[2])
        
        for d in datasets:
            path = os.path.join(obj.root_path,obj.dataset_prefix+str(d),obj.array_folder)
            if not sessions:
                sess_paths = glob.glob(os.path.join(path,'Session_*'))
                sess_paths.sort()
                sessions = [int(x.split('_')[-1]) for x in sess_paths]
            for s in sessions:
                if not files:
                    files = []
                    files_paths = glob.glob(os.path.join(path,f'Session_{s}','*.wav'))
                    files_paths.sort()
                    for file in files_paths:
                        files.append(file.split(os.sep)[-1].split('_M')[-1].split('.wav')[0])
                for f in files:
                    c['dataset'].append(d)
                    c['session'].append(s)
                    c['file'].append(f)
                files = []
        cases = pd.DataFrame(c)
        return cases
                        
    def get_VAD(obj):
        # DESCRIPTION: returns the vad of all sources over time
        # *** OUTPUTS ***
        # vads          (ndarray) VAD binary matrix [total_sources x total_frames] 
        # t             (ndarray) time frames in sec [total_frames x 1]
        # source_IDs    (ndarray) source/participant IDs [total_sources x 1]
        vads = np.zeros((obj.total_sources,obj.total_frames), dtype='int')
        t= obj.t
        source_IDs = np.arange(obj.total_sources)+1
        data = pd.read_csv(obj.vad_file)
        cols = list(data.keys())
        IDs = [n for n in range(1,obj.total_sources+1) if any(s for s in cols if str(n) in s)]
        data = data.to_numpy()  # (nFrame,4) [index time azi el] in degress
        for ID in IDs:
            j = cols.index('ID %d talking'%(ID))
            vads[ID-1,:]=data[:,j]
        return vads, t, source_IDs
        
    def plot_VAD(obj):
        # DESCRIPTION: plots the VADs of all sources over over time
        vads, t, source_IDs =  obj.get_VAD()
        nSrc, nFrame = vads.shape
        plt.figure()
        plt.pcolormesh(t,source_IDs,vads, shading='auto', cmap='binary', vmin = 0, vmax = 1)
        plt.xlabel('[sec]')
        plt.ylabel('Participant ID')
        plt.grid(linestyle = ':')
        plt.title('Dataset: %d, Session: %d, File: %s' % (obj.dataset,obj.session,obj.file))    
        