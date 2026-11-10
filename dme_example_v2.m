% DME v2 from MATLAB: 3D drift estimation on a simulated dataset (units: pixels, pixels, microns).
% Uses the Python package through MATLAB's Python interface (R2022a or newer), see matlab/dme_estimate_v2.m.
%
% One-time setup: point MATLAB to a Python environment with PyTorch (and optionally the dme package), e.g.
%   pyenv('Version', 'C:\Users\me\miniconda3\envs\smlm\python.exe', 'ExecutionMode', 'OutOfProcess')

addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab'))

n_frames = 4000;
fov_width = 200;
loc_error = [0.1, 0.1, 0.03];       % px, px, um
n_sites = 400;
on_prob = 0.1;

rng(1)
drift_true = cumsum([0.001, 0, 0] + 0.02 * randn(n_frames, 3), 1);
drift_true = drift_true - mean(drift_true, 1);

% binding sites that blink on and off; each localization is shifted by the drift of its frame
sites = [fov_width * rand(n_sites, 2), 2 * rand(n_sites, 1) - 1];
[f, s] = find(rand(n_frames, n_sites) < on_prob);
framenum = f - 1;                                   % frames start at 0
positions = sites(s, :) + drift_true(f, :) + loc_error .* randn(numel(f), 3);
crlb = repmat(loc_error, numel(f), 1);
fprintf('%d localizations\n', numel(f));

% DME v2: chooses the temporal resolution itself, learned temporal prior
tic
[drift, set1, set2] = dme_estimate_v2(positions, framenum, crlb);
fprintf('DME v2: %.1f s\n', toc);
% algorithm of the 2021 paper, spline knot every 20 frames
drift_v1 = dme_estimate_v2(positions, framenum, crlb, 'Method', 'v1', 'FramesPerBin', 20, 'EstimatePrecision', false);

err = @(d) sqrt(mean((d - mean(d, 1) - drift_true).^2, 1));
fprintf('DME v2: RMS error vs. true drift = %s (px, px, um)\n', mat2str(err(drift), 3));
fprintf('DME v1: RMS error vs. true drift = %s (px, px, um)\n', mat2str(err(drift_v1), 3));
fprintf('split-half precision estimate (v2): %s\n', mat2str(sqrt(mean((set1 - mean(set1, 1) - set2 + mean(set2, 1)).^2, 1)), 3));

corrected = positions - drift(framenum + 1, :);

labels = {'x [px]', 'y [px]', 'z [um]'};
figure
for i = 1:3
    subplot(3, 1, i)
    plot(drift_true(:, i), 'k', 'LineWidth', 2); hold on
    plot(drift(:, i)); plot(drift_v1(:, i), '--');
    ylabel(labels{i})
end
legend('true drift', 'DME v2', 'DME v1')
xlabel('frame')
