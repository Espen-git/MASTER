%% Path
image_dir = 'C:\Users\espen\Documents\Skole\MASTER\images\';

%% 4 (hyperechoic)
b_data_ML4 = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test4');
save_path = strcat(image_dir, 'b_data_ML4.mat');
save(save_path,'b_data_ML4');

%% 5 hypoechoic/hyperechoic (hyperechoic)
b_data_ML5 = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test5(Two_images)');
save_path = strcat(image_dir, 'b_data_ML5.mat');
save(save_path,'b_data_ML5');

%% 6 hypoechoic (hyperechoic)
b_data_ML6 = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test6(hypoechoic)');
save_path = strcat(image_dir, 'b_data_ML6.mat');
save(save_path,'b_data_ML6');

%% 7 hyperechoic (hypoechoic)
b_data_ML7 = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hypoechoic', 'Alpinion', 2, 'Ria_Test7(hyperechoic)');
save_path = strcat(image_dir, 'b_data_ML7.mat');
save(save_path,'b_data_ML7');

%% 8 Verasonics (hyperechoic)
b_data_ML8_hyperechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test8(Verasonics)');
save_path = strcat(image_dir, 'b_data_ML8_hyperechoic.mat');
save(save_path,'b_data_ML8_hyperechoic');
%% 8 Verasonics (Verasonics)
b_data_ML8_Verasonics = use_modified_capon_minimum_variance('Verasonics_P2-4_parasternal_long_small', 'Verasonics', 2, 'Ria_Test8(Verasonics)');
save_path = strcat(image_dir, 'b_data_ML8_Verasonics.mat');
save(save_path,'b_data_ML8_Verasonics');

%% 9 Verasonics (hyperechoic)
b_data_ML9_hyperechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test9(Verasonics-complex_network)');
save_path = strcat(image_dir, 'b_data_ML9_hyperechoic.mat');
save(save_path,'b_data_ML9_hyperechoic');
%% 9 Verasonics (Verasonics)
b_data_ML9_Verasonics = use_modified_capon_minimum_variance('Verasonics_P2-4_parasternal_long_small', 'Verasonics', 2, 'Ria_Test9(Verasonics-complex_network)');
save_path = strcat(image_dir, 'b_data_ML9_Verasonics.mat');
save(save_path,'b_data_ML9_Verasonics');

%% 10 Verasonics (hyperechoic)
b_data_ML10_hyperechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test10(Verasonics-complex_network2)');
save_path = strcat(image_dir, 'b_data_ML10_hyperechoic.mat');
save(save_path,'b_data_ML10_hyperechoic');
%% 10 Verasonics (Verasonics)
b_data_ML10_Verasonics = use_modified_capon_minimum_variance('Verasonics_P2-4_parasternal_long_small', 'Verasonics', 2, 'Ria_Test10(Verasonics-complex_network2)');
save_path = strcat(image_dir, 'b_data_ML10_Verasonics.mat');
save(save_path,'b_data_ML10_Verasonics');

%% 11 Verasonics (hyperechoic)
b_data_ML11_hyperechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test11(Verasonics-complex_network_4_frames)');
save_path = strcat(image_dir, 'b_data_ML11_hyperechoic.mat');
save(save_path,'b_data_ML11_hyperechoic');
%% 11 Verasonics (Verasonics)
b_data_ML11_Verasonics = use_modified_capon_minimum_variance('Verasonics_P2-4_parasternal_long_small_1_frame', 'Verasonics', 2, 'Ria_Test11(Verasonics-complex_network_4_frames)');
save_path = strcat(image_dir, 'b_data_ML11_Verasonics.mat');
save(save_path,'b_data_ML11_Verasonics');


%% MV
b_data_MV_hyperechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 0, 'Ria');
save_path = strcat(image_dir, 'b_data_MV_hyperechoic.mat');
save(save_path,'b_data_MV_hyperechoic');
b_data_MV_hypoechoic = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hypoechoic', 'Alpinion', 0, 'Ria');
save_path = strcat(image_dir, 'b_data_MV_hypoechoic.mat');
save(save_path,'b_data_MV_hypoechoic');
b_data_MV_Verasonics = use_modified_capon_minimum_variance('Verasonics_P2-4_parasternal_long_small', 'Verasonics', 0, 'Ria');
save_path = strcat(image_dir, 'b_data_MV_Verasonics.mat');
save(save_path,'b_data_MV_Verasonics');


%% DAS Verasonics
% data location
url='http://ustb.no/datasets/';      % if not found downloaded from here
data_path = 'C:\Users\espen\Documents\Skole\MASTER\code\data\';
local_path = strcat(data_path, 'Verasonics_P2-4_parasternal_long_small', '\'); % location of example data
addpath(local_path);

% check if the file is available in the local path or downloads otherwise
tools.download(strcat('Verasonics_P2-4_parasternal_long_small', '.uff'), url, local_path);
channel_data = uff.read_object([local_path, strcat('Verasonics_P2-4_parasternal_long_small', '.uff')],'/channel_data');

depth_axis = linspace(0e-3,110e-3,1024).';
        azimuth_axis = zeros(channel_data.N_waves,1);
        for n = 1:channel_data.N_waves
            azimuth_axis(n) = channel_data.sequence(n).source.azimuth;
        end
        scan = uff.sector_scan('azimuth_axis',azimuth_axis,'depth_axis',depth_axis);

mid = midprocess.das();
mid.channel_data = channel_data;
mid.scan = scan;
mid.dimension = dimension.both();
mid.transmit_apodization.window = uff.window.none;
mid.receive_apodization.window = uff.window.none;
b_data_DAS_Verasonics = mid.go();

save_path = strcat(image_dir, 'b_data_DAS_Verasonics.mat');
save(save_path,'b_data_DAS_Verasonics');

%% DAS Alpinion
% First image
url='http://ustb.no/datasets/';      % if not found downloaded from here
data_path = 'C:\Users\espen\Documents\Skole\MASTER\code\data\';
local_path = strcat(data_path, 'Alpinion_L3-8_CPWC_hyperechoic_scatterers', '\'); % location of example data
addpath(local_path);
% check if the file is available in the local path or downloads otherwise
tools.download(strcat('Alpinion_L3-8_CPWC_hyperechoic_scatterers', '.uff'), url, local_path);
channel_data = uff.read_object([local_path, strcat('Alpinion_L3-8_CPWC_hyperechoic_scatterers', '.uff')],'/channel_data');

% Scan
scan = uff.linear_scan();
scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512)';
scan.z_axis = linspace(5e-3,50e-3,512)';

% DAS first image
mid = midprocess.das();
mid.channel_data = channel_data;
mid.scan = scan;
mid.dimension = dimension.both();
mid.transmit_apodization.window = uff.window.none;
mid.receive_apodization.window = uff.window.none;
b_data_DAS_Alpinion_hyperechoic = mid.go();

save_path = strcat(image_dir, 'b_data_DAS_Alpinion_hyperechoic.mat');
save(save_path,'b_data_DAS_Alpinion_hyperechoic');

% Second image
url='http://ustb.no/datasets/';      % if not found downloaded from here
data_path = 'C:\Users\espen\Documents\Skole\MASTER\code\data\';
local_path = strcat(data_path, 'Alpinion_L3-8_CPWC_hypoechoic', '\'); % location of example data
addpath(local_path);
% check if the file is available in the local path or downloads otherwise
tools.download(strcat('Alpinion_L3-8_CPWC_hypoechoic', '.uff'), url, local_path);
channel_data = uff.read_object([local_path, strcat('Alpinion_L3-8_CPWC_hypoechoic', '.uff')],'/channel_data');

% DAS second image
mid = midprocess.das();
mid.channel_data = channel_data;
mid.scan = scan;
mid.dimension = dimension.both();
mid.transmit_apodization.window = uff.window.none;
mid.receive_apodization.window = uff.window.none;
b_data_DAS_Alpinion_hypoechoic = mid.go();

save_path = strcat(image_dir, 'b_data_DAS_Alpinion_hypoechoic.mat');
save(save_path,'b_data_DAS_Alpinion_hypoechoic');

