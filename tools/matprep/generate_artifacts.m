%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Script for reproducibly generating MATLAB PREP artifacts via CI %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Author: Austin Hurst


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 0: Initialize MATLAB environment for running PREP %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialize pipeline settings
addpath('config');
settings;

% Load EEGLAB and MATLAB PREP
addpath(eeglab_path);
addpath(genpath(matprep_path));
eeglab();

% Install BIOSIG package for reading EDF/BDF files
plugin_askinstall('Biosig', [], true);
warning('off');

% Print version information to CI
fprintf('\n\nVersion Info:\n');
fprintf([' - MATLAB: ' version '\n']);
fprintf([' - EEGLAB: ' eeg_getversion() '\n']);
fprintf([' - PREP: ' getPrepVersion '\n\n']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1: Load raw data into EEGLAB and prepare for PREP %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Actually read in data
pretty_header(['Processing ' filename]);
EEG = pop_biosig(filename);
EEG = eeg_checkset(EEG);
EEG.name = filename;

% Update PREP parameters using channel count and sample rate
eeg_channels = 1:EEG.nbchan;
params.referenceChannels = eeg_channels;
params.evaluationChannels = eeg_channels;
params.rereferencedChannels = eeg_channels;
params.detrendChannels = eeg_channels;
params.lineNoiseChannels = eeg_channels;
params.lineFrequencies = powerline_hz:powerline_hz:EEG.srate/2;

% Fix channel names so that montage loads correctly
for n = eeg_channels
    % Strip trailing '.'s from the channel names
    tmp = strrep(EEG.chanlocs(n).labels, '.', '');
    % Fix channel capitalization (i.e., all-caps except z and sometimes p)
    tmp = strrep(upper(tmp), 'Z', 'z');
    tmp = strrep(tmp, 'FP', 'Fp');
    % Write fixed label back to EEG object
    EEG.chanlocs(n).labels = tmp;
end

% Add montage to dataset
EEG = pop_chanedit(EEG, 'lookup', montage_path);
EEG = eeg_checkset(EEG);

% Create bad-by-NaN channels
ch_names = {EEG.chanlocs.labels};
for ch = 1:numel(bad_by_nan)
    ch_idx = find(matches(ch_names, bad_by_nan{ch}));
    EEG.data(ch_idx, 5) = NaN;
end

% Create bad-by-flat channels
for ch = 1:numel(bad_by_flat)
    ch_idx = find(matches(ch_names, bad_by_flat{ch}));
    EEG.data(ch_idx, :) = EEG.data(ch_idx, :) * 1e-13;
end

% Create bad-by-dropout channels
signal_len = length(EEG.data);
for ch = 1:numel(bad_by_dropout)
    ch_idx = find(matches(ch_names, bad_by_dropout{ch}));
    % Make 2nd and 4th quarters of channel recording flat
    EEG.data(ch_idx, floor(signal_len / 4):floor(signal_len / 2)) = 0;
    EEG.data(ch_idx, floor(3 * signal_len / 4):end) = 0;
end

% Prepare for PREP (copy/pasted from PREP internals)
EEG.etc.noiseDetection = ...
    struct(...
        'name', params.name, ...
        'version', getPrepVersion, ...
        'originalChannelLabels', [], ...
        'errors', [], ...
        'boundary', [], ...
        'detrend', [], ...
        'lineNoise', [], ...
        'reference', [], ...
        'postProcess', [], ...
        'interpolatedChannelNumbers', [], ...
        'removedChannelNumbers', [], ...
        'stillNoisyChannelNumbers', [], ...
        'fullReferenceInfo', false ...
    );
EEG.data = double(EEG.data);   % Don't monkey around -- get into double
EEG.etc.noiseDetection.originalChannelLabels = {EEG.chanlocs.labels};
defaults = getPrepDefaults(EEG, 'general');
[params, errors] = checkPrepDefaults(params, params, defaults);
if ~isempty(errors)
    error('prepPipeline:GeneralDefaultError', ['|' sprintf('%s|', errors{:})]);
end
defaults = getPrepDefaults(EEG, 'boundary');
[boundaryOut, errors] = checkPrepDefaults(params, struct(), defaults);
EEG.etc.noiseDetection.boundary = boundaryOut;

% Save a copy of the data prior to running PREP
save_set(EEG, [artifact_dir sep '1_matprep_raw']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 2: Actually run all components of the PREP pipeline %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remove trend prior to cleanline
fprintf('\n\n=== Removing Trends Pre-CleanLine... ===\n\n');
[EEGNew, detrend] = removeTrend(EEG, params);
EEG.etc.noiseDetection.detrend = detrend;
defaults = getPrepDefaults(EEG, 'detrend');
params = checkPrepDefaults(detrend, params, defaults);
save_set(EEGNew, [artifact_dir sep '2_matprep_removetrend']);

% Perform cleanline on data and put trend back
fprintf('\n\n=== Performing CleanLine... ===\n\n');
[EEGClean, lineNoise] = removeLineNoise(EEGNew, params);
EEG.etc.noiseDetection.lineNoise = lineNoise;
lineChannels = lineNoise.lineNoiseChannels;
EEG.data(lineChannels, :) = ...
    EEG.data(lineChannels, :) ...
    - EEGNew.data(lineChannels, :) ...
    + EEGClean.data(lineChannels, :);
clear EEGNew;
save_set(EEG, [artifact_dir sep '3_matprep_cleanline']);

% Get detrended data prior to rereferencing & save a copy
fprintf('\n\n=== Get Detrended Signal Pre-Reference... ===\n\n');
[EEGNew, detrend] = removeTrend(EEG, params);
save_set(EEGNew, [artifact_dir sep '4_matprep_pre_reference']);
clear EEGNew;

% Perform robust re-referencing on data
fprintf('\n\n=== Performing Robust Referencing... ===\n\n');
[EEG, referenceOut] = performReference(EEG, params);
save_set(EEG, [artifact_dir sep '5_matprep_post_reference']);

% Save detailed internal PREP details to a .mat
prep_info = EEG.etc.noiseDetection;
prep_info.reference = referenceOut;
prep_info.fullReferenceInfo = true;
prep_info.interpolatedChannelNumbers = referenceOut.interpolatedChannels.all;
prep_info.stillNoisyChannelNumbers = ...
    referenceOut.noisyStatistics.noisyChannels.all;
save([artifact_dir sep 'matprep_info'], 'prep_info');

% Print out first-pass NoisyChannels results for CI
fprintf('\nInitial findNoisyChannels Results:\n');
noisy_orig = referenceOut.noisyStatisticsOriginal.noisyChannels;
ch_names = prep_info.originalChannelLabels;
bad_types = fieldnames(noisy_orig);
for n = 1:numel(bad_types)
    bad_type = bad_types{n};
    chan_idx = noisy_orig.(bad_type);
    fprintf([' - ' bad_type ' = [']);
    for ch = 1:numel(chan_idx)
        if (ch > 1)
            fprintf(' ');
        end
        fprintf(ch_names{chan_idx(ch)});
    end
    fprintf(']\n');
end
fprintf('\n');


pretty_header([filename ' complete!']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define helper functions at the bottom of the file %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function for printing pretty headers/footers in MATLAB scripts
function [] = pretty_header(content);
    header = ['### ' content ' ###'];
    header_pad = join(repmat('#', 1, strlength(header)), '');
    fprintf(['\n' header_pad '\n']);
    fprintf([header '\n']);
    fprintf([header_pad '\n\n']);
end

% Function for saving EEG data to a single .set file
function [] = save_set(EEG, outpath);
    pop_saveset(EEG, 'filename', outpath, 'savemode', 'onefile');
end
