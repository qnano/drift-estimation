% main script for matlab implemention (v1)

clear all
warning('off')
addpath(genpath('./matlab'))          
%  
%% User inputs for simulation of data (3D)
usecuda = false;                           % Cuda implementation (bool)
n_frames = 500;                         % Number of frames in data 
fov_width = 200;                        % Field of view for simulation
drift_mean = [0.001,0,0];               % Mean drift in simulation
drift_stdev = [0.02,0.02,0.02];         % Std in drift simulation
loc_error = [0.1,0.1,0.03];             % Pixel, pixel, um
on_prob = 0.1;                          % On probability

%% User inputs for simulation of data (2D)
% usecuda = true;                           % Cuda implementation (bool)
% n_frames = 200;                         % Number of frames in data 
% fov_width = 200;                        % Field of view for simulation
% drift_mean = [0.001,0];               % Mean drift in simulation
% drift_stdev = [0.02,0.02];         % Std in drift simulation
% loc_error = [0.1,0.1];             % Pixel, pixel, um
% on_prob = 0.1;                          % On probability

%% User inputs RCC drift computation
timebins = 10;
zoom = 2;
sigma = 1;
maxpairs = 1000;

%% User inputs for drift estimation - !keep data types alive!
coarse_est = true;                      % Coarse drift estimation (bool)
precision_est = true;                  % Precision estimation (bool)
coarse_frames_per_bin = int32(10);      % Number of bins for coarse est. (int32)
framesperbin = int32(1);                % Number of frames per bin (int32)
maxneighbors_coarse = int32(1000);   % Max neighbors for coarse and precision est. (int32)
maxneighbors_regular = int32(1000);     % Max neighbors for regular est. (int32)
coarseSigma= single([0.2,0.2,0.2]);     % Localization precision for coarse estimation (single/float)                     
max_iter_coarse = int32(1000);          % Max iterations coarse est. (int32)
max_iter = int32(10000);                % Max iterations (int32)
gradientstep = single(1e-6);            % Gradient (single/float)

%% Simulation data

% Ground truth drift trace
drift_trace = [];
for i = 1:n_frames
    drift_trace = [drift_trace; normrnd(drift_mean, drift_stdev)];
end
drift_trace = cumsum(drift_trace,1);
drift_trace = drift_trace - mean(drift_trace, 1);
[localizations,framenum] = smlm_simulation(drift_trace, fov_width, loc_error, 200, n_frames, on_prob);
clrb = loc_error;

%% RCC computation
if size(drift_trace,2) == 3
    %RCC 3D
    [drift_xyz] = rcc3D(localizations, framenum, timebins, zoom, sigma, maxpairs,  usecuda);
else
    [drift_xyz] = rcc(localizations, framenum, timebins, zoom, sigma, maxpairs, usecuda);
    
end
%% Drift estimation
drift = dme_estimate(localizations, framenum, clrb, drift_xyz, usecuda, coarse_frames_per_bin, ...
    framesperbin, maxneighbors_coarse,maxneighbors_regular, coarseSigma, max_iter_coarse,max_iter, gradientstep, precision_est );

%% Plot
h= figure(1)
for i = 1:size(drift_trace,2)
    subplot(size(drift_trace,2),1,i);
    plot(1:1:n_frames, drift(:,i)-0.2, 'LineWidth', 2);
    hold on
    plot(1:1:n_frames, drift_xyz(:,i)+0.2, 'LineWidth', 2);
    hold on
    plot(1:1:n_frames, drift_trace(:,i), 'LineWidth', 2);
    legend('Estimated drift (DME)', 'Estimated drift (RCC)', 'True drift') ;
end




