function [drift, drift_set1, drift_set2] = dme_estimate_v2(positions, framenum, crlb, varargin)
%DME_ESTIMATE_V2  Drift estimation with DME v2, through the Python package (needs MATLAB R2022a or newer).
%
%   [drift, set1, set2] = dme_estimate_v2(positions, framenum, crlb)
%   [drift, set1, set2] = dme_estimate_v2(..., 'FramesPerBin', 100, 'Method', 'v1', ...)
%
%   positions : N x 2 or N x 3 localizations (e.g. x, y in pixels, z in microns)
%   framenum  : N x 1 frame numbers, starting at 0
%   crlb      : N x D localization precision (same units as positions), or a 1 x D vector
%
%   drift     : F x D drift, row f+1 is frame f. Corrected positions: positions - drift(framenum + 1, :)
%   set1/set2 : drift estimated from two halves of the data (only if 'EstimatePrecision' is true);
%               rms(set1 - set2) indicates the precision.
%
%   Options (name-value):
%     'FramesPerBin'      []: DME v2 chooses the temporal resolution and uses the learned temporal prior (default).
%                         A number: cubic spline with one knot per FramesPerBin frames (needed for 'v1').
%     'Method'            'v2' (default) or 'v1' (algorithm of the 2021 paper)
%     'EstimatePrecision' true (default) / false
%     'NumFrames'         number of frames (default max(framenum) + 1)
%     'Device'            'cuda', 'cpu' or [] (default: the GPU if available)
%     'Verbose'           true (default) / false
%
%   One-time setup: a Python environment with PyTorch and the dme package
%   (pip install git+https://github.com/qnano/drift-estimation.git), selected in MATLAB with
%       pyenv('Version', 'C:\path\to\env\python.exe', 'ExecutionMode', 'OutOfProcess')
%   Out-of-process mode avoids conflicts between the libraries of MATLAB and PyTorch. If the dme package is not
%   installed, the repository this file is in is used.

p = inputParser;
p.addParameter('FramesPerBin', []);
p.addParameter('Method', 'v2');
p.addParameter('EstimatePrecision', true);
p.addParameter('NumFrames', []);
p.addParameter('Device', []);
p.addParameter('Verbose', true);
p.parse(varargin{:});
o = p.Results;

dme = import_dme();
D = size(positions, 2);
if isvector(crlb) && numel(crlb) == D
    crlb = repmat(reshape(crlb, 1, D), size(positions, 1), 1);
end
if isempty(o.NumFrames), o.NumFrames = double(max(framenum)) + 1; end

kw = pyargs('framesperbin', none_or(o.FramesPerBin, @(v) int32(v)), ...
            'method', char(o.Method), ...
            'estimatePrecision', logical(o.EstimatePrecision), ...
            'numframes', int32(o.NumFrames), ...
            'device', none_or(o.Device, @char), ...
            'verbose', logical(o.Verbose));
r = dme.dme_estimate(py.numpy.array(single(positions)), py.numpy.array(int32(framenum(:))), ...
                     py.numpy.array(single(crlb)), kw);

if o.EstimatePrecision
    drift = double(r{1});
    sets = r{2};
    drift_set1 = double(sets{1});
    drift_set2 = double(sets{2});
else
    drift = double(r);
    drift_set1 = [];
    drift_set2 = [];
end
end


function dme = import_dme()
try
    dme = py.importlib.import_module('dme');
catch
    repo = fileparts(fileparts(mfilename('fullpath')));     % matlab/.. = repository root
    sys = py.importlib.import_module('sys');
    sys.path.insert(int32(0), repo);
    dme = py.importlib.import_module('dme');
end
end


function v = none_or(x, conv)
if isempty(x)
    v = py.None;
else
    v = conv(x);
end
end
