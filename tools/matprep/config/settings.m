%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Settings file for MATLAB PREP pipeline %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Important filenames and paths

sep = '/';

filename = 'S004R01.edf';
montage_name = 'standard-10-5-cap385.elp';
package_dir = 'deps';
artifact_dir = 'artifacts';

eeglab_path = [package_dir sep 'EEGLAB'];
matprep_path = [package_dir sep 'PREP' sep 'PrepPipeline'];
montage_dir = [eeglab_path sep 'plugins' sep 'dipfit' sep 'standard_BESA'];
montage_path = [montage_dir sep montage_name];


% Initialize PREP parameters

powerline_hz = 60;

params = struct();
params.name = filename;
params.detrendType = 'high pass';
params.detrendCutoff = 1;
params.referenceType = 'robust';
params.keepFiltered = false;


% Channels to make bad by various criteria

bad_by_nan = {'FC5'};
bad_by_flat = {'FC3'};
bad_by_dropout = {'Fpz'};
