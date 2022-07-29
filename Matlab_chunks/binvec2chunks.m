function out = binvec2chunks(in)
%splits a binary vector into start/stop chunks of 1's segments
%in : 1D vector
%out: Nx2 mateix [start stop] indices of N chunks
in = [0;in(:);0];
dd = diff(in);
i1 = find(dd==1);
i2 = find(dd==-1) - 1;
out = [i1(:) i2(:)];
end

