function [drift_updated, scores_updated, num_iterations, image] = MinEntropyDriftEstimate(pos, clrb,framenum,numspots,maxit,drift,framesperbin,gradientstep ...
    ,maxdrift,scores,flags,maxneighbors, usecuda, draw_gaus, varargin)

if isempty(varargin)
   
    img_input = single(1);
    img_w = int32(1);
    img_h = int32(1);
    spotlist = single(1);
    numspots_gaus = int32(1);
    
end

if ~isempty(varargin)
    
    
    img_input = varargin{1};
    img_w = varargin{2};
    img_h = varargin{3};
    spotlist = varargin{4};
    numspots_gaus = varargin{5};
    
end


if usecuda

[drift_updated, scores_updated, num_iterations, image] = dme_cuda(pos',clrb',framenum',numspots',maxit',drift',framesperbin',gradientstep' ...
    ,maxdrift',scores',flags',maxneighbors', draw_gaus', img_input', img_w, img_h, spotlist', numspots_gaus');
else
    [drift_updated, scores_updated, num_iterations, image] = dme_cpu(pos',clrb',framenum',numspots',maxit',drift',framesperbin',gradientstep' ...
    ,maxdrift',scores',flags',maxneighbors', draw_gaus', img_input', img_w, img_h, spotlist', numspots_gaus');

end
scores_updated = scores_updated(1:num_iterations)';
drift_updated = drift_updated';
image = image';


end

