function drift = dme_estimate(position, framenumbers, crlb_initial, drift_guess, usecuda, coarse_frames_per_bin, ...
    framesperbin, maxneighbors_coarse,maxneighbors_regular, coarseSigma, max_iter_coarse,max_iter, gradientstep, precision_est )


% initialize variables
pos = single(position);
crlb = single(crlb_initial);
framenum = int32(framenumbers);
drift = single(drift_guess);
maxdrift = single(0); % not used
numspots = int32(size(pos,1));


% compute sets for precision estimate
[~, split_axis] = max(var(pos(:,1:2), 0, 1));
split_value = median(pos(:,split_axis));
set1 = pos(:,split_axis) > split_value;
set2 = ~set1;
% -------------------------------------------------------------------------------------------------------

% Coarse estimation
fprintf('Computing coarse drift estimation...\n');
[drift, scores, iter] = call_dme_lib('dme', usecuda, pos, coarseSigma, framenum, numspots, drift, coarse_frames_per_bin,gradientstep ...
    ,maxdrift,maxneighbors_coarse, max_iter_coarse);

% regular drift estimation
fprintf('Computing regular drift estimation...\n');
[drift, scores_updated, num_iterations] = call_dme_lib('dme', usecuda, pos, crlb, framenum, numspots, drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);


% precision estimate
if precision_est
    fprintf('Computing precision set 1 drift estimation...\n');
    [drift_set1, scores_updated_set1, num_iterations_set1] = call_dme_lib('dme', usecuda, pos(set1, :), crlb(set1,:), framenum(set1), int32(size(pos(set1, :),1)), drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);
    
    fprintf('Computing precision set 2 drift estimation...\n');
    [drift_set2, scores_updated_set2, num_iterations_set2] = call_dme_lib('dme', usecuda, pos(set2, :), crlb(set2,:), framenum(set2), int32(size(pos(set2, :),1)), drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);
    
    drift_set1 = drift_set1 - mean(drift_set1);
    drift_set2 = drift_set2 - mean(drift_set2);
    L = min(length(drift_set1), length(drift_set2));
    diff = drift_set1(1:L,:)-drift_set2(1:L,:);
    rmsd = sqrt(mean(diff.^2));
    fprintf(['RMSD of drift traces on split dataset = ', num2str(rmsd)] )
end


drift = drift - mean(drift);
    
    

