function [outputArg1,outputArg2] = smlm_simulation(drift_trace, fov_width, loc_error, n_sites, n_frames, on_prob)
fprintf('Simulate data ... \n')
if size(drift_trace,2) == 2
    binding_sites = rand(n_sites, 2);
    binding_sites(:,(1:2)) = binding_sites(:,(1:2))*fov_width;

else
    binding_sites = rand(n_sites, 3);
    binding_sites(:,(1:2)) = binding_sites(:,(1:2))*fov_width;
    binding_sites(:, 3) = binding_sites(:, 3)*2 -1;


end

% typical 2D acquisition with small Z range and large XY range 

localizations = [];
framenum = [];


for i = 1:n_frames
    on = logical(binornd(1, on_prob, [n_sites,1]));
    locs = binding_sites(on, :);
    % add localization error
    temp_loc_error = [];
    for j = 1:size(locs,1)
        temp_loc_error(j,:) = normrnd(0, loc_error);
    end
    locs = locs + drift_trace(i,:) + temp_loc_error;
    framenum = [framenum ones(1, length(locs))*i];
    localizations = [localizations; locs];
      
end
outputArg1 = localizations;
outputArg2 = framenum;
end

