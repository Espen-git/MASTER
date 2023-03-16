%%
close all;
clear all;

% Data path
path = 'C:\Users\espen\Documents\Skole\MASTER\Data\';
addpath(path);

% Chose file
filename = 'Verasonics_P2-4_parasternal_long_small.uff';
%filename = 'PICMUS_experiment_contrast_speckle.uff';
%filename = 'FieldII_P4_point_scatterers.uff';
%filename = 'Alpinion_L3-8_CPWC_hypoechoic.uff';
%filename = 'Alpinion_L3-8_CPWC_hyperechoic_scatterers.uff';
% Import file
channel_data = uff.read_object([path, filename],'/channel_data');

%% Define the UFF sector_scan
depth_axis=linspace(0e-3,110e-3,1024).';                
azimuth_axis=linspace(channel_data.sequence(1).source.azimuth,...
    channel_data.sequence(end).source.azimuth,channel_data.N_waves)';
scan=uff.sector_scan('azimuth_axis',azimuth_axis,'depth_axis',depth_axis);

%% Define the UFF linear_scan
scan=uff.linear_scan();
scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512).';
scan.z_axis = linspace(1e-3,50e-3,512).';

%% BEAMFORMER 
mid=midprocess.das();                                
mid.channel_data=channel_data;
mid.dimension = dimension.transmit();
mid.scan=scan;
mid.transmit_apodization.window=uff.window.scanline;
mid.receive_apodization.window=uff.window.boxcar;
b_data = mid.go();
%%
mv = postprocess.capon_minimum_variance();

mv.dimension = dimension.receive();
mv.scan = scan;
mv.receive_apodization = mid.receive_apodization;
mv.transmit_apodization = mid.transmit_apodization;

mv.channel_data = channel_data;
mv.input = b_data;

mv.L_elements = 10;
mv.K_in_lambda = 1
mv.regCoef = 1;

b_data_mv = mv.go()

%% DAS for comparison
mid=midprocess.das();                                
mid.channel_data=channel_data;
mid.dimension = dimension.both();
mid.scan=scan;
mid.transmit_apodization.window=uff.window.scanline;
mid.receive_apodization.window=uff.window.boxcar;
b_data_das = mid.go();

%%
b_data_mv.plot([],['MV'],[80],[],[],[],[],'dark');      % Display
b_data_das.plot([],['DAS'],[80],[],[],[],[],'dark');      % Display



