function [VA] = SPEAR_get_voice_activity(filename)
% Main read function for head-tracking data from SPEAR


% filename = '/Volumes/SPEAR_SSD3/SPEAR/InitialRelease/Extra/Train/Dataset_2/VAD/Session_1/vad_D2_S1_M03.csv';
A = readmatrix(filename);
A(1,:) = [];


VAD = zeros(7,size(A,1));

D4_check = 0;
if contains(filename, 'Dataset_4')
    D4_check = 1;
end

if size(A,2)==6-D4_check*2
    if ~D4_check
        VAD(1,:) = A(:,6);
        VAD(2,:) = A(:,3);
    end
    VAD(4,:) = A(:,4-D4_check);
    VAD(6,:) = A(:,5-D4_check);
elseif size(A,2)==7-D4_check*2
    if ~D4_check
        VAD(1,:) = A(:,7);
        VAD(2,:) = A(:,3);
    end
    VAD(3,:) = A(:,4-D4_check);
    VAD(5,:) = A(:,5-D4_check);
    VAD(7,:) = A(:,6-D4_check);
elseif size(A,2)==8-D4_check*2
    if ~D4_check
        VAD(1,:) = A(:,8);
        VAD(2,:) = A(:,3);
    end
    VAD(3,:) = A(:,4-D4_check);
    VAD(4,:) = A(:,5-D4_check);
    VAD(6,:) = A(:,6-D4_check);
    VAD(7,:) = A(:,7-D4_check);
end

VA.VA = VAD;

end