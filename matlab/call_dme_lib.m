function [varargout] = call_dme_lib(string_func, usecuda, varargin) 

[path,name,ext]=fileparts(mfilename('fullpath'));

if usecuda
    fn= [path '\..\bin\release\dme_cuda.dll'];
else
    fn= [path '\..\bin\release\dme_cpu.dll'];
end

hfile = [path '\..\dme\DME\DriftEstimation.h'];

loadlibrary(fn, hfile, 'alias', 'dme');



if strcmp(string_func, 'gauss')
    %% Draw gaussians
    % Get data
    img_input = varargin{1};
    img_input = libpointer('singlePtr', img_input' );
    img_w = varargin{2};
    img_h = varargin{3};
    spotlist = varargin{4};
    numspots_gaus = varargin{5};
    
    % CDLL_EXPORT void Gauss2D_Draw(float* image, int imgw, int imgh, float* spotList, int nspots);
    calllib('dme', 'Gauss2D_Draw', img_input, int32(img_w'), int32(img_h'), single(spotlist'), int32(numspots_gaus'));
    varargout{1} = img_input.value' ;
    


elseif strcmp(string_func, 'dme')
    %% DME
    % Get data
    coords = varargin{1};
    crlb = varargin{2};
    spotFramenum = varargin{3};
    numspots = varargin{4};
    drift = varargin{5};
    framesperbin = varargin{6};
    gradientstep = varargin{7};
    maxdrift = varargin{8};
    maxneighbors = varargin{9};
    max_iter = varargin{10};
    flags = 0;
    
    spotFramenum = spotFramenum -1;
    if usecuda
       flags = bitor(flags,2); 
    end
    if size(coords,2) == 3
        flags = bitor(flags,1);
    end
    
    if length(crlb) == size(coords,2)
        flags = bitor(flags, 4); % always, no variable CRLB implementation
    else
        assert(isequal(size(crlb), size(coords)), 'CRLB matrix should either be [1, ndims] or [npos, ndims]');
    end
    
    drift = libpointer('singlePtr', drift' );
    
    %CDLL_EXPORT IDriftEstimator* DME_CreateInstance(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
    %	float* drift, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors);
    
    inst = calllib('dme', 'DME_CreateInstance', single(coords'), single(crlb'), int32(spotFramenum'), int32(numspots'), drift, int32(framesperbin') ...
                    , single(gradientstep'), single(maxdrift'), int32(flags), int32(maxneighbors));
    

    statusbufsize = 200;
    scores = zeros(max_iter);
    score = single(0);
    
    cr='';
    for i = 1:max_iter
         %CDLL_EXPORT int DME_Step(IDriftEstimator* estimator, char* status_msg, int status_max_length, float* score, float* drift_estimate);
        [r, inst_, statusmsg, score, drift_] = calllib('dme', 'DME_Step', inst, blanks(statusbufsize), statusbufsize, score, drift);
        scores(i) = score;

        fprintf(cr);
        fprintf('%s', statusmsg);
        cr = repmat('\b',1,length(statusmsg));

        if r == 0
            break
        end
    end
    fprintf('%s\n',statusmsg);
    
    calllib('dme', 'DME_Close', inst);
    
    varargout{1} = drift.value';
    varargout{2} = scores(1:i);
    varargout{3} = i;
    
else
    error('Error: wrong input')
end


unloadlibrary('dme');

end
