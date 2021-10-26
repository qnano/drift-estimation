

% need to start using relative paths here
fn= 'C:\dev\drift-estimation\dme\x64\Release\dme-cpu.dll';
hfile = 'C:\dev\drift-estimation\dme\DME\DriftEstimation.h';

loadlibrary(fn, hfile, 'alias', 'dme');

libfunctions('dme')



%CDLL_EXPORT IDriftEstimator* DME_CreateInstance(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
%	float* drift, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors);

%// Drift estimation step. Zero pointers can be passed to status_msg, score, and drift_estimate if not needed
%CDLL_EXPORT int DME_Step(IDriftEstimator* estimator, char* status_msg, int status_max_length, float* score, float* drift_estimate);

%CDLL_EXPORT void DME_Close(IDriftEstimator* estim);

N=2000;
nframes=50;
pos = rand(N, 2) * 50;
crlb = ones(N,2) * 0.15;
framenum = randi(nframes, N);

drift = zeros(nframes, 2);

framesPerBin = 10;
gradientStep = 1e-5;
maxdrift=2; 
inst = calllib('dme', 'DME_CreateInstance', pos, crlb, framenum, N, drift, framesPerBin, gradientStep, maxdrift, 4, 1000);
disp(inst);

pscore = libpointer('single');
status = blanks(100);
%estim = drift*0;
for i = 1:1
    calllib('dme', 'DME_Step', inst, status, 100, 0, 0);% status, length(status), pscore, 0);
    fprintf('%d', i);
end

calllib('dme', 'DME_Close', inst);

unloadlibrary('dme');

