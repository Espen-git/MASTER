%% Load 
clear all;
close all;

image_dir = 'C:\Users\espen\Documents\Skole\MASTER\images\';

% ML images
b_data_ML4 = load(strcat(image_dir,'b_data_ML4.mat')).b_data_ML4;
b_data_ML5 = load(strcat(image_dir,'b_data_ML5.mat')).b_data_ML5;
b_data_ML6 = load(strcat(image_dir,'b_data_ML6.mat')).b_data_ML6;
b_data_ML7 = load(strcat(image_dir,'b_data_ML7.mat')).b_data_ML7;

b_data_ML8_hyperechoic = load(strcat(image_dir,'b_data_ML8_hyperechoic.mat')).b_data_ML8_hyperechoic;
b_data_ML8_Verasonics = load(strcat(image_dir,'b_data_ML8_Verasonics.mat')).b_data_ML8_Verasonics;

b_data_ML9_hyperechoic = load(strcat(image_dir,'b_data_ML9_hyperechoic.mat')).b_data_ML9_hyperechoic;
b_data_ML9_Verasonics = load(strcat(image_dir,'b_data_ML9_Verasonics.mat')).b_data_ML9_Verasonics;

b_data_ML10_hyperechoic = load(strcat(image_dir,'b_data_ML10_hyperechoic.mat')).b_data_ML10_hyperechoic;
b_data_ML10_Verasonics = load(strcat(image_dir,'b_data_ML10_Verasonics.mat')).b_data_ML10_Verasonics;

b_data_ML11_hyperechoic = load(strcat(image_dir,'b_data_ML11_hyperechoic.mat')).b_data_ML11_hyperechoic;
b_data_ML11_Verasonics = load(strcat(image_dir,'b_data_ML11_Verasonics.mat')).b_data_ML11_Verasonics;

% MV images
b_data_MV_hyperechoic = load(strcat(image_dir,'b_data_MV_hyperechoic.mat')).b_data_MV_hyperechoic;
b_data_MV_hypoechoic = load(strcat(image_dir,'b_data_MV_hypoechoic.mat')).b_data_MV_hypoechoic;
b_data_MV_Verasonics = load(strcat(image_dir,'b_data_MV_Verasonics.mat')).b_data_MV_Verasonics;

% DAS images
b_data_DAS_Alpinion_hyperechoic = load(strcat(image_dir,'b_data_DAS_Alpinion_hyperechoic.mat')).b_data_DAS_Alpinion_hyperechoic;
b_data_DAS_Alpinion_hypoechoic = load(strcat(image_dir,'b_data_DAS_Alpinion_hypoechoic.mat')).b_data_DAS_Alpinion_hypoechoic;
b_data_DAS_Verasonics = load(strcat(image_dir,'b_data_DAS_Verasonics.mat')).b_data_DAS_Verasonics;


%% Plots
scale = max(b_data_MV_hyperechoic.data(:));
dynamic_range_alpinion = [-60 0];

plot_alpinion_image(b_data_ML4, scale, 'ML_4', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML5, scale, 'ML_5', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML6, scale, 'ML_6', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML7, scale, 'ML_7', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML8_hyperechoic, scale, 'ML_8_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML9_hyperechoic, scale, 'ML_9_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML10_hyperechoic, scale, 'ML_10_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_ML11_hyperechoic, scale, 'ML_11_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_MV_hyperechoic, scale, 'MV_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_MV_hypoechoic, scale, 'MV_hypoechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_DAS_Alpinion_hyperechoic, scale, 'DAS_hyperechoic', dynamic_range_alpinion);
plot_alpinion_image(b_data_DAS_Alpinion_hypoechoic, scale, 'DAS_hypoechoic', dynamic_range_alpinion);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dynamic_range_verasonics = [-70 -15];

plot_verasonics_image(b_data_ML8_Verasonics, 'ML_8_Verasonics', dynamic_range_verasonics);
plot_verasonics_image(b_data_ML9_Verasonics, 'ML_9_Verasonics', dynamic_range_verasonics);
plot_verasonics_image(b_data_ML10_Verasonics, 'ML_10_Verasonics', dynamic_range_verasonics);
plot_verasonics_image(b_data_ML11_Verasonics, 'ML_11_Verasonics', dynamic_range_verasonics);
plot_verasonics_image(b_data_MV_Verasonics, 'MV_Verasonics', dynamic_range_verasonics);
plot_verasonics_image(b_data_DAS_Verasonics, 'DAS_Verasonics', dynamic_range_verasonics);

%%

%%
function plot_alpinion_image(b_data, scale, name, dynamic_range)
    font = 15;
    pos = [1000 918 560-180 420];
    
    img = reshape(b_data.data, [512,512]);
    x = b_data.scan.x_axis*1000;
    z = b_data.scan.z_axis*1000;
    
    figure;
    imagesc(x, z, db(abs(img./scale)));
    ylabel(['z [mm]']); xlabel('x [mm]');
    colormap gray;
    caxis(dynamic_range);
    hcb = colorbar;
    hcb.Title.String = 'dB';
    %title(name);
    set(gca,'fontsize',font);
    set(gcf, 'position',pos);
    saveas(gcf, strcat('images\', name, '.svg'))
end

function plot_verasonics_image(b_data, name, dynamic_range)
    font = 15;
    pos = [976 437 600 700];

    b_data.plot([],[],[],[],[],[],[],[]);
    ylabel(['z [mm]']); xlabel('x [mm]');
    colormap gray; 
    caxis(dynamic_range); 
    hcb = colorbar;
    hcb.Title.String = 'dB';
    set(gca,'fontsize',font);
    set(gcf, 'position',pos);
    saveas(gcf, strcat('images\', name, '.svg'))
end

% plot(figure_handle_in,in_title,dynamic_range,compression,indeces,frame_idex,spatial_units,mode)
    %   figure_handle   Handle to the figure to plot to (default: none)
    %   title           Figure title (default: none)
    %   dynamic_range   Displayed dynamic range (default: 60 dB)
    %   compression     String specifying compression type: 'log','none','sqrt' (default: 'log')
    %   indeces         Pair of integers [nrx ntx] indicating receive and transmit events (default: [])
    %   indeces         Tripler of integers [nrx ntx frame] indicating which receive and transmit and frame must be plotted (default: [])