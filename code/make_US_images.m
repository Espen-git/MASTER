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


%% DAS
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
b_data_DAS = mid.go();

%% Load 
image_dir = 'C:\Users\espen\Documents\Skole\MASTER\images\';
b_data_ML8_hyperechoic = load(strcat(image_dir,'b_data_ML8_hyperechoic.mat')).b_data_ML8_hyperechoic;
b_data_ML9_hyperechoic = load(strcat(image_dir,'b_data_ML9_hyperechoic.mat')).b_data_ML9_hyperechoic;
b_data_ML10_hyperechoic = load(strcat(image_dir,'b_data_ML10_hyperechoic.mat')).b_data_ML10_hyperechoic;


b_data_MV_hyperechoic = load(strcat(image_dir,'b_data_MV_hyperechoic.mat')).b_data_MV_hyperechoic;
%% Scan
name = 'Alpinion_L3-8_CPWC_hyperechoic_scatterers';
url='http://ustb.no/datasets/';      % if not found downloaded from here
data_path = 'C:\Users\espen\Documents\Skole\MASTER\code\data\';
local_path = strcat(data_path, name, '\'); % location of example data
addpath(local_path);
% check if the file is available in the local path or downloads otherwise
tools.download(strcat(name, '.uff'), url, local_path);
channel_data = uff.read_object([local_path, strcat(name, '.uff')],'/channel_data');

scan = uff.linear_scan();
scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512).';
scan.z_axis = linspace(1e-3,50e-3,512).';

%% Plots
x = scan.x_axis*1000;
z = scan.z_axis*1000;
font = 10;
pos = [1000 918 560-180 420];

%ML_4 = reshape(b_data_ML4.data, [512,512]);
%ML_5 = reshape(b_data_ML5.data, [512,512]);
%ML_6 = reshape(b_data_ML6.data, [512,512]);
%ML_7 = reshape(b_data_ML7.data, [512,512]);
ML_8_hyper = reshape(b_data_ML8_hyperechoic.data, [512,512]);
%ML_8_verasonics = reshape(b_data_ML8_Verasonics.data, [101,1024]);
ML_9_hyper = reshape(b_data_ML9_hyperechoic.data, [512,512]);
%ML_9_verasonics = reshape(b_data_ML9_Verasonics.data, [101,1024]);
ML_10_hyper = reshape(b_data_ML10_hyperechoic.data, [512,512]);
%ML_10_verasonics = reshape(b_data_ML10_Verasonics.data, [101,1024]);
MV_hyper = reshape(b_data_MV_hyperechoic.data, [512,512]);
%MV_hypo = reshape(b_data_MV_hypoechoic.data, [512,512]);
%MV_verasonics = reshape(b_data_MV_Verasonics.data, [101,1024]);

scale = max(MV_hyper(:));

%plot_US_image(ML_4, x, z, scale, font, pos, 'ML\_4');
%plot_US_image(ML_5, x, z, scale, font, pos, 'ML\_5');
%plot_US_image(ML_6, x, z, scale, font, pos, 'ML\_6');
%plot_US_image(ML_7, x, z, scale, font, pos, 'ML\_7');
plot_US_image(ML_8_hyper, x, z, scale, font, pos, 'ML\_8\_hyper');
plot_US_image(ML_9_hyper, x, z, scale, font, pos, 'ML\_9\_hyper');
plot_US_image(ML_10_hyper, x, z, scale, font, pos, 'ML\_10\_hyper');
%plot_US_image(MV_hyper, x, z, scale, font, pos, 'MV\_hyper');
%plot_US_image(MV_hypo, x, z, scale, font, pos, 'MV\_hypo');
%b_data_ML9_Verasonics.plot();


function plot_US_image(b_data, x, z, scale, font, pos, name)
    figure()
    imagesc(x, z, db(abs(b_data./scale)));
    colormap gray; caxis([-60 0]); colorbar;
    title(name);
    set(gca,'fontsize',font);
    set(gcf, 'position',pos);
end




