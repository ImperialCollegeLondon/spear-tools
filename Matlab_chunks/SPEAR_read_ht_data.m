function [ht] = SPEAR_read_ht_data(filename)
% Main read function for head-tracking orientation data from SPEAR


% filename = '/Volumes/SPEAR_SSD3/SPEAR/InitialRelease/Extra/Train/Dataset_2/Reference_PosOri/Session_1/00/refOri_D2_S1_M00_ID4.csv';
A = readmatrix(filename);
A(1,:) = [];

ht.t=A(:,2);
ht.x=A(:,3);
ht.y=A(:,4);
ht.z=A(:,5);
ht.w=A(:,6);

end