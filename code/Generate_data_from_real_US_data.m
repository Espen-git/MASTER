% Make R matrices form real US data
%% 
clear all; close all;

% data location
url='http://ustb.no/datasets/';      % if not found downloaded from here
local_path = [ustb_path(),'/data/']; % location of example data
addpath(local_path);

% Choose dataset
%filename='Verasonics_P2-4_parasternal_long_small.uff';
filename='Alpinion_L3-8_CPWC_hyperechoic_scatterers';
%filename='Alpinion_L3-8_CPWC_hypoechoic.uff';
%filename='Alpinion_L3-8_FI_hyperechoic_scatterers.uff';
%filename='Alpinion_L3-8_FI_hypoechoic.uff';
% check if the file is available in the local path or downloads otherwise
tools.download(strcat(filename, '.uff'), url, local_path);

% check if the file is available in the local path or downloads otherwise
tools.download(strcat(filename, '.uff'), url, local_path);
channel_data = uff.read_object([local_path, strcat(filename, '.uff')],'/channel_data')

% For Alpion
%uff_file=uff(filename)
%channel_data = uff_file.read('/channel_data');

%% Verasonics P2-4
% Define the scan
depth_axis=linspace(0e-3,110e-3,1024).';
azimuth_axis=zeros(channel_data.N_waves,1);
for n=1:channel_data.N_waves
    azimuth_axis(n) = channel_data.sequence(n).source.azimuth;
end
scan=uff.sector_scan('azimuth_axis',azimuth_axis,'depth_axis',depth_axis);

%% Alpinion L3-8
scan=uff.linear_scan();
scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512).';
scan.z_axis = linspace(1e-3,50e-3,512).';

%% PICMUS
scan=uff.linear_scan()
scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512)';
scan.z_axis = linspace(5e-3,50e-3,512)';

%%
% transmit beamforming (before MV)
mid = midprocess.das();
mid.channel_data = channel_data;
mid.scan = scan;
mid.dimension = dimension.transmit();
mid.transmit_apodization.window = uff.window.none;
mid.receive_apodization.window = uff.window.none;
b_data_transmit = mid.go();

%% MV
post = capon_R_out();
post.channel_data = channel_data;
post.input = b_data_transmit;
post.dimension = dimension.receive();
post.scan = scan;
%post.transmit_apodization.window = mid.receive_apodization;
%post.receive_apodization.window = mid.transmit_apodization;

post.L_elements = 16; % subarray size
post.K_in_lambda = 1; % temporal averaging factor
post.regCoef = 0; % regularization factor

post.go(filename);

%%
%save('R_Alpinion_L3-8_CPWC_hypoechoic.mat','R');

%% MV
%b_data_mv.plot([],['MV'],[80],[],[],[],[],'dark');      % Display

%%

%mid=midprocess.das();
%mid.channel_data=channel_data;
%mid.scan=scan;
%mid.dimension=dimension.both()
%mid.transmit_apodization.window=uff.window.none;
%mid.receive_apodization.window=uff.window.none;
%b_data_DAS = mid.go();

%% DAS
%b_data_DAS.plot([],['DAS'],[80],[],[],[],[],'dark');      % Display

