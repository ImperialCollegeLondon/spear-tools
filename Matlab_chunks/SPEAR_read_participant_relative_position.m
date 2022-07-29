function out = SPEAR_read_participant_relative_position(ht_file,participant_ID)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mode='FRL_JSON';
nFrames=1200;
wearer_participant_ID=2;
ht = SPEAR_read_ht_data(ht_file); % load ori ID 2

ht_file_pos = strrep(ht_file,'refOri','refPos');
loc0=SPEAR_read_target_data(ht_file_pos); % load pos ID2

t=loc0.t;
l0=[loc0.x loc0.y loc0.z];
q=quaternion(ht.w,ht.x,ht.y,ht.z);
out.t = t;

ht_file_pos_IDX = strrep(ht_file_pos,'ID2',sprintf('ID%d',participant_ID));
loc = SPEAR_read_target_data(ht_file_pos_IDX); % load pos IDX
l = [loc.x loc.y loc.z];
l = l - l0;

cc=zeros(length(t),3);
for frame = 1:length(t)
    rotated_xyz=rotatepoint(q(frame),[1 0 0;0 1 0;0 0 1]);
    x=dot(rotated_xyz(1,:),l(frame,:));
    y=dot(rotated_xyz(2,:),l(frame,:));
    z=dot(rotated_xyz(3,:),l(frame,:));
    cc(frame,:)=[x y z];
end
out.x = griddedInterpolant(t,cc(:,1),'linear');
out.y = griddedInterpolant(t,cc(:,2),'linear');
out.z = griddedInterpolant(t,cc(:,3),'linear');  
end

