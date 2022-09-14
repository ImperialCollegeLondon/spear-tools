# Imperial College London. 2022
# Baseline Processor class for SPEAR Challenge

# Version history: date/author/updates:
# 2022 May - Sina Hafezi - first version inclusing baseline beamformer (Isotropic-MVDR/Superdirective) & all required utility functions
# 2022 Jul - Sina Hafezi - moved pos/ht interpolation from dataset class to here
# 2022 Aug - Sina Hafezi - faster stft processing (one less loop)

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
import librosa
import numpy.matlib
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import interp1d

class SPEAR_Processor:
    out_chan = [2] # (list) channels at which to get enhanced audio (indexing from 1)
    
    total_sensor = [] # (int) total number of sensors
    dirs = [] # (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
    azi = [] # (ndarray) azimuth in radian [nAzimuth x 1]
    ele = [] # (ndarray) elevation in radian [nElevation x 1]
    
    IR = [] # (ndarray) AIR directory [nSample x nDirection x nChan]
    ATF = [] # (ndarray) fft of IR [nFreq x nDirection x nChan] (Acoustic Transfer Function)
    f = [] # (ndarray) frequency axis
    t = [] # (ndarray) time axis
    
    mics = [] # (list) mics subset
    fs = [] # (int) sample rate in Hz
    nChan = [] # (int) number of sensors in mics subset
    nDir = [] # (int) number of measrued directions in AIR/ATF
    nFreq = [] # (int) number of frequencies
    nSample = [] # (int) number of samples in time
    nOut = [] # (int) number of output channels
    
    # STFT params (dict)
    stft_params = {'window': 'hann', 'winL': 16e-3, 'stepL': 8e-3}
    nfft = [] # FFT size in sample
    noverlap = [] # step size in sample
    
    target_pos_interp = []  # (SciPy 1D-interpolant) cartesian position interpolant function returning array of (x y z) as ndarray for requested time(s)
    target_dir_ind = [] # (list) hold index of target doa over time
    
    R_iso = [] # diffuse covariance matrix (used in superdirective beamformer - baseline method)
    w_conj = []  # pre-calculated beamforming conjugate weights for all directions [nChan x nFreq x nDir x nOut]
    w_stft = [] # beamforming conjugate weights in stft domain [nChan x nFreq x nFrame x nOut]
    diag_load_mode = 'cond' # type of diagonal loading. 'cond': limit maximum condition number. 'const': add a constant identity. [] to disable diag loading
    diag_load_val = int(1e2) # max condition number for 'cond' mode, constant value for 'const' mode
    
    # settings for beam pattern visualisation
    beampattern = {'az_res': 5, 'el_res': 10}
    
    def __init__(obj,air,fs = [],mics = [],out_chan = out_chan):
        # DESCRIPTION: Intakes array's AIR and initialises the parameters
        # *** INPUTS *** 
        # air  (dict) dictionary of AIR (output of get_all_AIRs() in dataset class)
        # fs (int) processing sample rate. [] for default
        # mics (list) list of mic subsets to be used for beamforming (indexing from 1). [] for all
        # out_chan (list) list of mic subsets at which the beamforming output are obtained (indexing from 1)

        obj.IR = air['IR'] # (ndarray) Impulse Responses [nSample x nDirection x nChan]
        fs0 = air['fs'] 
        obj.total_sensor = air['nChan']
        obj.dirs = air['directions'] #  (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
        obj.azi = air['azi'] # (ndarray) [nAzimuth x 1]
        obj.ele = air['ele'] # (ndarray) [nElevation x 1]
        if not fs:
            obj.fs = fs0
        else:
            obj.fs = int(fs)
        if not mics:
            obj.mics = list(np.arange(obj.total_sensor)+1)
        else:
            obj.mics = mics
        obj.nChan = len(obj.mics)
        
        obj.IR = obj.IR[:,:,list(np.array(obj.mics)-1)] # keeping the IRs for required mics subset
        nSample, nDir, nChan = obj.IR.shape
        if obj.fs!=fs0: # resamplilng IRs if needed
            obj.IR = np.transpose(obj.IR) # [nChan x nDir x nSample ]
            obj.IR = obj.IR.reshape((-1,nSample)) # [nChan*nDir x nSample ]
            obj.IR = librosa.resample(obj.IR,orig_sr=fs0,target_sr=obj.fs) # [nChan*nDir x nSample2 ]
            obj.IR = obj.IR.reshape((nChan,nDir,-1))  # [nChan x nDir x nSample2 ]
            obj.IR = np.transpose(obj.IR) # [nSample2 x nDir x nChan]
            nSample, nDir, nChan = obj.IR.shape
        
        obj.nSample = nSample
        obj.nChan = nChan
        obj.nDir = nDir
        
        obj.set_out_channels(out_chan)
        obj.prepare_ATF()
        obj.prepare_iso_weights()
    
    def set_out_channels(obj,c):
        # DESCRIPTION: update the requested output channels for beamformer
        # *** INPUTS ***
        # c   (list) list of mic subsets at which the beamforming output are obtained (indexing from 1)
        obj.out_chan = c 
        obj.nOut = len(c)
        
    def set_target_pos(obj,target_pos):
        # DESCRIPTION: update the target position interpolant function for the requested target
        # *** INPUTS ***
        # target_pos   (ndarray) cartesian position over time samples (t x y z) as [total_frames x 4]
        obj.target_pos_interp = interp1d(target_pos[:,0],target_pos[:,1:],kind='previous',axis=0,fill_value='extrapolate')
        
    def set_diag_loading(obj,mode='cond',val=int(1e5)):
        # DESCRIPTION: set the settings for diagonal loading of covariance matrix
        # *** INPUTS ***
        # mode      (str) 'cond': based on maximum condition limiting. 'const': based on adding a fixed constant 
        # val       (float) maximum condition number if mode=='cond' or weight of identity matrix added if mode=='const'
        obj.diag_load_mode = mode
        obj.diag_load_val = val
        obj.calculate_iso_cov()
        obj.prepare_iso_weights()
        
    
    def set_stft_params(obj,window='hann', winL=16e-3, stepL=8e-3):
        # DESCRIPTION: set the settings for STFT
        # *** INPUTS ***
        # window (str)  type of window
        # winL (float) time frame size in sec
        # stepL (float) time frame step in sec
        obj.stft_params['window'] = window
        obj.stft_params['winL'] = winL
        obj.stft_params['stepL'] = stepL
        obj.prepare_ATF()
    
    def prepare_ATF(obj):
        # DESCRIPTION: obtains ATF from AIR using FFT
        obj.nfft = round(obj.stft_params['winL'] * obj.fs)
        obj.noverlap = round(obj.stft_params['stepL'] * obj.fs)
        
        # AIR to ATF (fft)
        obj.ATF = rfft(obj.IR,n=obj.nfft,axis=0)
        obj.f = rfftfreq(obj.nfft,1/obj.fs)
        obj.nFreq = len(obj.f)
        
        obj.calculate_iso_cov()
    
    def calculate_iso_cov(obj):
        # DESCRIPTION: calculates the isotropic diffuse covariance matrix (for superdirective/Iso-MVDR beamformer)
        # ATF [nFreq x nDir x nChan]
        w_quad = obj.get_quadrature_weights() # [nDir x 1]
        w_quad = np.matlib.tile(w_quad,(obj.nFreq,obj.nChan,obj.nChan,1)) # [nFreq x nChan x nChan x nDir]
        w_quad = np.transpose(w_quad,axes=[0,3,1,2]) # [nFreq x nDir x nChan x nChan]
        obj.R_iso = obj.ATF[:,:,:,None] @ np.conj(obj.ATF[:,:,None,:]) # [nFreq x nDir x nChan x nChan]
        obj.R_iso = obj.R_iso * w_quad # quadrature weighting
        obj.R_iso = np.squeeze(np.sum(obj.R_iso,axis = 1))  # [nFreq x nChan x nChan]
        if obj.diag_load_mode:
            # diag loading is requested
            for fi in range(obj.nFreq):
                obj.R_iso[fi,:,:] = obj.diag_load_cov(np.squeeze(obj.R_iso[fi,:,:]))
        
        
    def diag_load_cov(obj,R):
        # DESCRIPTION: diagonal loading of covariance matrix
        # *** INPUTS ***
        # R (ndarray)  covariance matrix [nChan x nChan]
        if obj.diag_load_mode=='const':
            R = R + obj.diag_load_val * np.eye(obj.nChan)
        else:
            cn0 = np.linalg.cond(R) # original condition number
            threshold = obj.diag_load_val 
            if cn0>threshold:
                ev = np.linalg.eig(R)[0] # eigenvalues only
                R = R + np.eye(obj.nChan) * (ev.max() - threshold * ev.min()) / (threshold-1)
        return R
                
    def get_quadrature_weights(obj):
        # DESCRIPTION: returns the quadrature weights for the elevations 
        # NOTE: These weights are needed to compensate for higher density of points closer to the poles due to uniform grid sampling of full-space
        inc = (np.pi/2)-obj.dirs[:,1] # [nDir x 1]
        Ninc = np.prod(obj.ele.shape)
        Nazi = np.prod(obj.azi.shape)
        m = np.arange(Ninc/2)
        m = m[:,None].T # (1,M)
        w = np.sum(np.sin(inc[:,None] * (2*m+1)) / (2*m+1),axis=1) * np.sin(inc) * 2 / (Ninc*Nazi) #(nDir,1)
        w = w / np.sum(w)
        return w
        
    def prepare_iso_weights(obj):
        # DESCRIPTION: calculates the beamformer weights for all directions and frequencies (to be called once and be used several)
        nFreq = obj.nFreq
        nChan = obj.nChan
        nOut = obj.nOut
        nDir = obj.nDir
        outChan_ind = np.array(obj.out_chan)-1
        obj.w_conj = np.zeros((nChan,nFreq,nDir,nOut),dtype=np.complex64) # conjugate weights        
        print_cycle=50 # number of interations to wait before updating progress print (keep it high as the loop is fast)
        for di in range(nDir):
            if (di%print_cycle==0):
                print('\r','Preparing weights %%%2.2f ' % (100*di/nDir),end='\r')
            h = np.squeeze(obj.ATF[:,di,:]) #  unnormalized RTF [nFreq x nChan]
            for fi in range(nFreq):
                R = np.squeeze(obj.R_iso[fi,:,:])  # [nChan x nChan]
                for oi in range(nOut):
                    rtf = h / h[:,outChan_ind[oi]].reshape(-1,1) # normalized RTF [nFreq x nChan]
                    obj.w_conj[:,fi,di,oi] = np.conj(obj.get_mvdr_weights(R,rtf[fi,:])) # [nChan x 1]
        print('\r','Preparing weights %%%2.2f - Done!' % (100),end='\n')         
       
    def get_angle_between(obj,ref,x,unit='radian'):
        # DESCRIPTION: calculates the angular separation between the directions (in either cartesian or DOA format)
        # *** INPUTS *** 
        # ref       (ndarray) reference direction(s) [1x2] or [Nx2] (azimuth,elevation) or [1x3] or [Nx3] (x,y,z)
        # x         (ndarray) subject direction(s) [Nx2] (azimuth,elevation) or [Nx3] (x,y,z)
        # unit      (str)   unit of DOA if ref and x inputs are DOAs 'radian' or 'degree'
        # *** OUTPUTS ***
        # a         (ndarray) angle between direction(s) [Nx1] in the same unit as input
        if ref.shape[0] != x.shape[0]:
            # there is one ref and multiple subject. repeat ref to match the number of subjects
            ref = np.matlib.repmat(ref,x.shape[0],1)
        if ref.shape[1]>2:
            # inputs are in cartesian. convert to spherical angles
            ref = np.array(obj.cart2sph(ref[:,0], ref[:,1], ref[:,2]))[[0,1],:].T # [N x 2] (az,el)
            x = np.array(obj.cart2sph(x[:,0], x[:,1], x[:,2]))[[0,1],:].T # [N x 2] (az,el)
        else:
            # inputs are in spherical angles. check unit
            if unit=='degree':
                ref = np.deg2rad(ref)
                x = np.deg2rad(x)
                
        # ref and x are now both in spherical angles (radian) and match in size
        # use Haversine formula to get angle between
        dlon = ref[:,0] - x[:,0] # azimuth differences
        dlat = ref[:,1] - x[:,1] # elevation differences 
        
        a = np.sin(dlat/2) ** 2 + np.cos(x[:,1]) * np.cos(ref[:,1]) * np.sin(dlon/2) ** 2
        a = 2 * np.arcsin(np.sqrt(a))
        if unit=='degree': # convert unit to match input unit
                a = np.rad2deg(a)
        return a
   
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
   
    def process_stft(obj,X,target_doas):
        # DESCRIPTION: applies superdirective beamforming in the STFT domain
        # *** INPUTS ***
        # X  (ndarray) signal in the stft domain [nChan x nFreq x nFrame]
        # target_doas  (ndarray) target DOA (azimuth,elevation) in radians [nFrame x 2]
        # *** OUTPUTS ***
        # Y  (ndarray) enhanced signal in the stft domain [nOut x nFreq x nFrame]
        # w  (ndarray) beamformer conjugate weights in the stft domain [nChan x nFreq x nFrame x nOut]
        nChan, nFreq, nFrame = X.shape # [nChan x nFreq x nFrame]
        nOut = len(obj.out_chan)
        w = np.zeros((nChan,nFreq,nFrame,nOut),dtype=np.complex64) # beamforning STFT conj weights [nChan x nFreq x nFrame x nOut]
        Y = np.zeros((nOut,nFreq,nFrame),dtype=np.complex64) # enhanced signal in STFT domain [nOut x nFreq x nFrame]
        obj.target_dir_ind = [0]*nFrame # initize the array length to store index of target direction per frame
        #print('Enhancement Processing')
        print_cycle=100 # number of interations to wait before updating progress print (keep it high as the loop is fast)
        # STFT beamforming
        for framei in range(nFrame):
            if (framei%print_cycle==0):
                print('\r','Processing %%%2.2f ' % (100*framei/nFrame),end='\r')
            dist = obj.get_angle_between(target_doas[framei,:],obj.dirs)
            di = np.where(dist == dist.min())[0][0] # nearest neighbour to target doa
            obj.target_dir_ind[framei]=di
            for oi in range(nOut):
                # read conj weights from pre-stored directory obj.w_conj = [nChan x nFreq x nDir x nOut]
                w[:,:,framei,oi] = obj.w_conj[:,:,di,oi]
                Y[oi,:,framei] = np.sum(X[:,:,framei] * w[:,:,framei,oi],axis=0)
        obj.w_stft = w            
        print('\r','Processing %%%2.2f - Done!' % (100),end='\n')
        return Y, w
    
    def get_mvdr_weights(obj,R,d):
        # DESCRIPTION: calculates the MVDR weights
        # *** INPUTS ***
        # R  (ndarray) covariance matrix [nChan x nChan]
        # d  (ndarray) steering vector or Relative Transfer Function (RTF) [nChan x 1]
        # *** OUTPUTS ***
        # w  (ndarray) beamformer conjugate weights in the stft domain  [nChan x 1]
        invRd = np.matmul(np.linalg.pinv(R),d)
        w = invRd/np.matmul(np.conj(d).T,invRd)
        return w
        
    def do_stft(obj,x):
        # DESCRIPTION: convert signal from time domain to STFT domain
        # *** INPUTS ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # *** OUTPUTS ***
        # X  (ndarray) signal in STFT domain [nChan x nFreq x nFrame]
        # f  (ndarray) frequency vector [nFreq x 1]
        # t  (ndarray) time vector [nFrame x 1]
        f, t, X = signal.stft(x, fs=obj.fs, window=obj.stft_params['window'], nperseg=round(obj.fs*obj.stft_params['winL']),noverlap=round(obj.fs*obj.stft_params['stepL']))
        obj.f = f
        obj.t = t
        return X, f, t
    
    def do_istft(obj,X):
        # DESCRIPTION: convert signal from time domain to STFT domain
        # *** INPUTS ***
        # X  (ndarray) signal in STFT domain [nChan x nFreq x nFrame]
        # *** OUTPUT ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # t  (ndarray) time vector [nFrame x 1]
        t, x = signal.istft(X, fs=obj.fs, window=obj.stft_params['window'], nperseg=round(obj.fs*obj.stft_params['winL']),noverlap=round(obj.fs*obj.stft_params['stepL']))
        return x, t
    
    def process_file(obj,in_file,out_file=[],start_t=0.0,duration=None):
        # DESCRIPTION: applies beamforming on an audio file and optionally write the enhanced file as an audio file
        # *** INPUTS ***
        # in_file  (str) path to the input audio file
        # out_file (str) optional path to the output audio file to write. [] value avoids writing file and only outputs the signal
        # *** OUTPUT ***
        # y  (ndarray) processed signal in the time doamin [nOut x nSample]
        # t  (ndarray) time vector [nFrame x 1]
        x, t = obj.read_audio(in_file,obj.fs,start_t,duration,obj.mics)
        y = obj.process_signal(x,start_t = start_t)
        if out_file:
            sf.write(out_file, y.T, obj.fs)
        return y, t
            
        
    def process_signal(obj,x,start_t = 0.0,target_doa=[]):
        # DESCRIPTION: applies beamforming on a time-doamin signal
        # *** INPUTS ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # start_t  (float) time in sec representing the start of the input signal (used for target DOA reading)
        # target_doa (ndarray) target DOA (azimuth,elevation) in radians [nFrame x 2]
        # *** OUTPUT ***
        # y  (ndarray) processed signal [nOut x nSample]
        X, _, t = obj.do_stft(x)
        if not target_doa:
            pos = obj.target_pos_interp(start_t+t)
            az, el, _ = obj.cart2sph(pos[:,0],pos[:,1],pos[:,2]) 
            target_doa = np.array([az,el]).T
        Y, _ = obj.process_stft(X,target_doa)
        y, _ = obj.do_istft(Y)
        return y
    
    
    def read_audio(obj,in_file,fs=None,start_t=0.0,duration=None,mics=[]):
        # DESCRIPTION: returns the time-domain audio signal
        # *** INPUTS ***
        # in_file  (str) full path to the audio file
        # fs       (int) optional - output sample rate in Hz, default: original sample rate
        # start_t   (float) optional - start time of segment in sec, default: beginning of the file
        # duration   (float) optional - duration of segment in sec, default: entirely to the end of the file
        # mics      (list) optional - mics/channels subset (indexing from 1), default: all channels
        # *** OUTPUTS ***
        # y   (ndarray) output array audio signal [nChan x nSample]
        # fs  (int)     output sample rate in Hz
        y, fs = librosa.load(in_file,sr=fs,mono=False,offset=start_t,duration=duration)
        if mics:
            y = y[np.array(mics)-1,:]
        return y, fs
     
    def get_beampattern_from_weights(obj,start_t=0.0,duration=[]):
        # DESCRIPTION: returns the time-domain audio signal
        # *** INPUTS ***
        # start_t   (float) time for a single frame or start time in sec of animated beam pattern
        # duration (float) duration time in sec for animated beam pattern (use 0 for single frame)
        # *** OUTPUTS ***
        # B     (ndarray) Beam pattern [nFreq x nDOA x nFrame]
        # f     (ndarray) frequency vector in Hz [nFreq x 1]
        # doas  (ndarray) DOAs axis in degrees [nDOA x 2] (azimuth,elevation). note: elevations are 0 (horizontal plane) 
        # t     (ndarray) time vector in sec [nFrame x 1]
        if isinstance(duration,list):
            duration = obj.t[-1]- obj.t[0]
        frame_ind = np.argmin(abs(obj.t[:,None] - np.array([start_t,start_t+duration])),axis=0) # start and stop frame [1,2]
        # horizontal plane directions
        az_step = 1
        azs = np.linspace(-180,180,int(360/az_step+1))
        eles = np.zeros(azs.shape)
        doas = np.array([azs,eles]).T # [nDOA x 2] degree
        nDOA = len(azs)
        doas_i = []
        for n in range(doas.shape[0]):
            dist = obj.get_angle_between(np.deg2rad(doas[n,:]),obj.dirs)
            doas_i.append(np.where(dist == dist.min())[0][0]) # nearest neighbour to target doa
        doas_i = np.array(doas_i)
        # obj.w_stft [nChan x nFreq x nFrame x nOut]
        out_chan_ind = 0
        w = np.squeeze(obj.w_stft[:,:,frame_ind[0]:frame_ind[1]+1,out_chan_ind]) # [nChan x nFreq x nFrame]
        if len(w.shape)==2:
            w = w[:,:,None,None] # [nChan x nFreq x 1 x 1]
        else:
            w = w[:,:,None,:] # [nChan x nFreq x 1 x nFrame]
        H = obj.ATF[:,doas_i,:,None]  # [nFreq x nDOA x nChan x 1]
        H = np.transpose(H,axes=[2,0,1,3]) # [nChan x nFreq x nDOA x 1]
        # H @ w : [nChan x nFreq x nDOA x 1] @ [nChan x nFreq x 1 x nFrame] -> [nChan x nFreq x nDOA x nFrame]
        B = np.squeeze(np.sum(H @ w,axis=0)) # [nFreq x nDOA x nFrame]
        
        # normalization by ATF at the target direction at every time frame (to get flat response at the target)
        target_dir_i = np.array(obj.target_dir_ind)[frame_ind[0]:frame_ind[1]+1] # [nFrame x 1]
        # ATF [nFreq x nDir x nChan]
        H_target = obj.ATF[:,target_dir_i,np.array(obj.out_chan[out_chan_ind])-1] # [nFreq x nFrame] 
        H_target = np.matlib.tile(H_target,(nDOA,1,1)) # [nDOA x nFreq x nFrame]
        H_target = np.transpose(H_target,axes=[1,0,2]) # [nFreq x nDOA x nFrame]
        if H_target.shape[2]==1:
            H_target = np.squeeze(H_target)
        
        B = B / H_target
        
        t = obj.t[frame_ind[0]:frame_ind[1]+1]
        f = obj.f
        return B, f, doas, t
        
    def plot_beampattern(obj,start_t=0.0,duration=0):
        # DESCRIPTION: plot single or animated beam pattern(s) 
        # *** INPUTS ***
        # start_t   (float) time for a single frame or start time in sec of animated beam pattern
        # duration (float) duration time in sec for animated beam pattern (use 0 for single frame)
        B, f, doas, t = obj.get_beampattern_from_weights(start_t=start_t,duration=duration)
        if duration==0:
            B = B[:,:,None]
            
        pos = obj.target_pos_interp(t)
        az, el, _ = obj.cart2sph(pos[:,0],pos[:,1],pos[:,2]) 
        az = np.rad2deg(az)
        
        # (nFreq,nDOA,nFrame)
        fig, ax = plt.subplots()
        for frame in range(B.shape[2]):
            plt.pcolormesh(doas[:,0],f,20*np.log10(abs(np.squeeze(B[:,:,frame]))), shading='auto', cmap='magma', vmin=-30, vmax=0)
            plt.title('Time: %2.2f s, Target Azimuth: %d deg' % (t[frame],az[frame]))
            plt.plot(np.ones((2,))*az[frame],f[np.array([0,-1])],':k')
            plt.ylabel('Freq [Hz]')
            plt.xlabel('Azimuth [deg]')
            plt.gca().invert_xaxis()
            plt.xticks(np.linspace(-180,180,9))
            plt.colorbar(label = '[dB]')
            #plt.pause(.1)
            plt.show()        