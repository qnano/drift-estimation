function drift = dme_estimate(position, framenumbers, crlb_initial, drift_guess, usecuda, coarse_frames_per_bin, ...
    framesperbin, maxneighbors_coarse,maxneighbors_regular, coarseSigma, max_iter_coarse,max_iter, gradientstep, precision_est )


% initialize variables
pos = single(position);
clrb = single(crlb_initial);
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
fprintf('\nComputing coarse drift estimation...\n');
[drift, scores, iter] = library_loader('dme', usecuda, pos, coarseSigma, framenum, numspots, drift, coarse_frames_per_bin,gradientstep ...
    ,maxdrift,maxneighbors_coarse, max_iter_coarse);
fprintf('\ncoarse drift estimation in %d iterations\n', iter);

% regular drift estimation
fprintf('\nComputing regular drift estimation...\n');
[drift, scores_updated, num_iterations] = library_loader('dme', usecuda, pos, coarseSigma, framenum, numspots, drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);
fprintf('\nRegular drift estimation in %d iterations\n', num_iterations);


% precision estimate
if precision_est
    fprintf('\nComputing precision set 1 drift estimation...\n');
    [drift_set1, scores_updated_set1, num_iterations_set1] = library_loader('dme', usecuda, pos(set1, :), coarseSigma, framenum(set1), int32(size(pos(set1, :),1)), drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);
    fprintf('\nset 1 precision estimate drift estimation in %d iterations\n', num_iterations_set1+1)
    
    fprintf('\nComputing precision set 2 drift estimation...\n');
    [drift_set2, scores_updated_set2, num_iterations_set2] = library_loader('dme', usecuda, pos(set2, :), coarseSigma, framenum(set2), int32(size(pos(set2, :),1)), drift, framesperbin,gradientstep ...
    ,maxdrift,maxneighbors_regular, max_iter);
    fprintf('\nset 2 precision estimate drift estimation in %d iterations\n', num_iterations_set2+1)
    
    drift_set1 = drift_set1 - mean(drift_set1);
    drift_set2 = drift_set2 - mean(drift_set2);
    L = min(length(drift_set1), length(drift_set2));
    diff = drift_set1(1:L,:)-drift_set2(1:L,:);
    rmsd = sqrt(mean(diff.^2));
    fprintf(['\nRMSD of drift traces on split dataset = ', num2str(rmsd)] )
end


drift = drift - mean(drift);
    
    

