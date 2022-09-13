%% Script to extract all valid chunks from SPEAR dataset
% and store its metadata in a csv file.
% Metrics will be calculated for each chunk in this dictionary. 

% Updates:
% 12/04/2022  Sina Hafezi     initial code that works based on EasyCom data structure
% xx/06/2022  Pierre Guiraud  Modify the paths to work on SPEAR data structure
% 24/08/2022  Pierre Guiraud  Modify the output CSV to add index and reference columns


clear
close all

% ------ SETTINGS ------
% inputs paths
j=1; % segment cntr
segments=struct();
tic;
for dataset_n = 1:4
    dataset = sprintf('Dataset_%d/', dataset_n);
    inpath.dataset='/media/spear/SPEAR_SSD1/SPEAR_v3/SPEAR'; %'/Volumes/HD FRL/data/EasyComDataset/Main';
    inpath.extra = 'Extra';
    inpath.main = 'Main';
    inpath.set = 'Dev';
    inpath.ref= [dataset, 'Reference_Audio']; %'Close_Microphone_Audio_Corrected';
    inpath.array=[dataset, 'Microphone_Array_Audio'];%'Glasses_Microphone_Array_Audio';
    inpath.vad=[dataset, 'VAD'];%'Speech_Transcriptions';
    inpath.pos=[dataset, 'Reference_PosOri'];%'Tracked_Poses';
    % inpath.pos=[dataset, 'Array_Orientation'];%'Tracked_Poses';
    
    outpath.path=pwd; % save path for output csv
    outpath.filename=sprintf('segments_%s', inpath.set);  % name for output csv
    dataset_id=sprintf('D%d', dataset_n); % dataset ID (value used in the associated column)
    if dataset_n==1, dataset_type='rec'; else, dataset_type='sim'; end % 'sim' or 'rec' (value used in the associated column)
    RT60=.645; % sec  (value used in the associated column)
    ATF_type='unknown'; % in case ATFs other than EasyCom were used (value used in the associated column)
    lev_ambient=nan; % in case level of oracle ambient is known (value used in the associated column)
    lev_sources=nan; % in case level of oracle sources is known (value used in the associated column)
    lev_sensor=nan;  % in case level of oracle sensor is known (value used in the associated column)
    
    
    refChan=2; % reference channel of array used for signal power
    server_ID=2; % participant ID of source wearing the glasses
    fs=48e3; % audio files sample rate
    fps=20; % OptiTrack frame rate
    nFrames=fps * 60; % total number of OptiTrack frames per file
    min_chunk_duration = .4; % seconds
    min_chunk_frames = round(min_chunk_duration*fps); % frames
    total_sources = 7; % maximum number of possible participants could be present
    
    if strcmp(inpath.set, 'Train')
        nSession_min = 1;
        nSession_max = 9;
    elseif strcmp(inpath.set, 'Dev')
        nSession_min = 10;
        nSession_max = 12;
    elseif strcmp(inpath.set, 'Eval')
        nSession_min = 13;
        nSession_max = 15;
    end
    
    % ------ ----- ------
    
    
    frame2sample = reshape(1:round(60*fs),round(fs / fps),[])';
    for session=    nSession_min:nSession_max
        sessionfolder=sprintf('Session_%d',session);
    
        files=dir(fullfile(inpath.dataset,inpath.extra,inpath.set,inpath.vad,sessionfolder,'*.csv'))
        nFile=length(files);
        for fi=1:nFile
            fprintf('** %s set - Dataset %d/4 - Session: %d/%d - File: %d/%d\n',inpath.set,dataset_n,session,nSession_max,fi,nFile);
            file=files(fi);
            fname=file.name(1:end-4);
            minute=file.name(end-5:end-4);
            str_file=file.name(5:end-4);
            VA=SPEAR_get_voice_activity(fullfile(file.folder,file.name));
            
            ht_file=fullfile(inpath.dataset,inpath.extra,inpath.set,inpath.pos,sessionfolder,minute,['refOri_', str_file, '_ID2.csv']);
    %         ht_file=fullfile(inpath.dataset,inpath.main,inpath.set,inpath.pos,sessionfolder,['ori', str_file, '.csv']);
            
            ht = SPEAR_read_ht_data(ht_file);
            
            [array, ~]=audioread(fullfile(inpath.dataset,inpath.main,inpath.set,inpath.array,sessionfolder,['array_', str_file, '.wav']));
            
            % going through each source as target
            srcs=dir(fullfile(inpath.dataset,inpath.extra,inpath.set,inpath.ref,sessionfolder,minute,sprintf('ref_%s_*.wav',str_file))); %['ref_', str_file, '*.wav']
            nSrc=length(srcs);
            chk_idx = 1;
            for si=1:nSrc
                sname=srcs(si).name;
                src_ID= str2double(sname(end-4)); %str2num(sname(find(sname=='_',1,'last')+1:find(sname=='.',1,'last')-1));
                
                [headset, ~]=audioread(fullfile(srcs(si).folder,srcs(si).name));
                % zero padding if headset signal is slightly shorter than array signal
                if size(headset,1)~=size(array,1)
                    headsettemp=headset;
                    headset=zeros(size(array,1),1);
                    headset(1:length(headsettemp))=headsettemp;
                end
                
                % frames with active target and inactive server
                valid_frames=(VA.VA(src_ID,:) & ~VA.VA(server_ID,:)); %[1 x nFrames]
                % identifying continuous chunks from binary vector of frames
                valid_chunks = binvec2chunks(valid_frames); % [nChunk x 2] (start_frame stop frames)
                valid_chunks(find(diff(valid_chunks,[],2)<min_chunk_frames),:)=[]; % removing very short chunks
                nChunk=size(valid_chunks,1);
                
                pos_interp = SPEAR_read_participant_relative_position(ht_file,src_ID);
                pos.t=pos_interp.t;
                pos.cart=[pos_interp.x(pos.t) pos_interp.y(pos.t) pos_interp.z(pos.t)];
                
                
                
                for ci=1:nChunk
                    frame1 = valid_chunks(ci,1);
                    frame2 = valid_chunks(ci,2);
                    sec1 = (frame1-1) / fps; % time @ the start of frame 1
                    sec2 = frame2 / fps;     % time @ the end of frame 2
                    sample1 = round(sec1 * fs)+1;
                    sample2 = round(sec2 * fs);
                    sample2 = min(sample2,size(array,1));
                    
                    active_src=find(sum(VA.VA(:,frame1:frame2),2)>0);
                    
                    tis=find(ht.t>=sec1 & ht.t<=sec2);
                    ht_q=[ht.w(tis) ht.x(tis) ht.y(tis) ht.z(tis)];
                    
                    tis=find(pos.t>=sec1 & pos.t<=sec2);
                    cart_in=pos.cart(tis,:);
                    mean_cart=mean(cart_in);
                    [az_mean, el_mean, ~]=cart2sph(mean_cart(1),mean_cart(2),mean_cart(3));
                    az_mean=wrapTo180(rad2deg(az_mean));
                    el_mean=wrapTo180(rad2deg(el_mean));
                    ang_dist = get_angle_between(mean_cart,cart_in); % deg
     
                    segments(j).global_index=j;
    
                    segments(j).dataset=dataset_id;
                    segments(j).session=session;
                    segments(j).minute=minute;
                    segments(j).file_index=fi;
                    segments(j).file_name=str_file;
                    segments(j).chunk_index=chk_idx;
                    chk_idx = chk_idx+1;
                    
                    segments(j).duration_frame=frame2-frame1+1;
                    segments(j).duration_sec=segments(j).duration_frame/fps;
                    
                    segments(j).frame_start=frame1;
                    segments(j).frame_stop=frame2;
                    segments(j).sec_start=sec1;
                    segments(j).sec_stop=sec2;
                    segments(j).sample_start=sample1;
                    segments(j).sample_stop=sample2;
                    
                    segments(j).target_ID=src_ID;
                    segments(j).num_active_sources=length(active_src);
                    
                    %segments(j).active_sources_ID=active_src;
                    for ssi = 1:total_sources
                        segments(j).(sprintf('vad_ID_%d',ssi)) = ismember(ssi,active_src);
                    end
                    
                    segments(j).sim_rec=dataset_type;
                    segments(j).rt60=RT60;
                    
                    %segments(j).target_doa_mean_deg=[az_mean el_mean];
                    segments(j).target_azi_deg = az_mean;
                    segments(j).target_ele_deg = el_mean;
                    segments(j).target_doa_dev_deg=mean(ang_dist);
                    
                    %segments(j).head_quart_mean=mean(ht_q);
                    mean_q = mean(ht_q);
                    segments(j).mean_quart_x = mean_q(1); % X
                    segments(j).mean_quart_y = mean_q(2); % Y
                    segments(j).mean_quart_z = mean_q(3); % Z
                    segments(j).mean_quart_w = mean_q(4); % W
                    
                    %segments(j).head_quart_std=std(ht_q);
                    std_q = std(ht_q);
                    segments(j).std_quart_x = std_q(1); % X
                    segments(j).std_quart_y = std_q(2); % Y
                    segments(j).std_quart_z = std_q(3); % Z
                    segments(j).std_quart_w = std_q(4); % W
                    
                    segments(j).target_dB=mag2db(rms(headset(sample1:sample2)));
                    segments(j).array_dB=mag2db(rms(array(sample1:sample2,refChan)));
                    
                    j=j+1;
                end
                
            end
            
           
        end
    end
end
toc
save(fullfile(outpath.path,sprintf('%s.mat',outpath.filename)),'segments','-v7.3');


%% csv write

fprintf('Creating the csv file \n')
ffs=fields(segments);
c=ffs(:)';
for j=1:length(segments)
    ct=cell(1,length(ffs));
    for ci=1:length(ffs)
        ct{ci}=segments(j).(ffs{ci});
    end
    c=[c;ct];
end
writecell(c,fullfile(outpath.path,sprintf('%s.csv',outpath.filename)));
fprintf('All done \n')
