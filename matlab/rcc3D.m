function [drift_xyz] = rcc3D(xyz, framenum, timebins, zoom, sigma, maxpairs, usecuda)

xy = xyz(:,1:2);
[drift_xy, ~, ~] = rcc(xy, framenum, timebins, zoom, sigma, maxpairs, usecuda);

sheared = xyz(:,1:2);
for i = 1:length(framenum)
    frame = framenum(i);
    sheared(i,:) = sheared(i,:) - drift_xy(frame,:);
end
sheared(:,2) = sheared(:,2)+ xyz(:,3);

[drift_sheared, ~,~]  = rcc(sheared, framenum, timebins, zoom, sigma, maxpairs, usecuda);

drift_xyz = zeros(length(drift_xy),3);

drift_xyz(:,1:2) = drift_xy;

drift_xyz(:,3) = drift_sheared(:,2);

drift_xyz = drift_xyz - mean(drift_xyz);

end

