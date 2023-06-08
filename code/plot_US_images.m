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


%% Alpinion Plots
scale = max(b_data_MV_hyperechoic.data(:));
dynamic_range_alpinion = [-60 0];
diff = false;

%plot_alpinion_image(b_data_ML4, scale, 'ML_4', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML5, scale, 'ML_5', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML6, scale, 'ML_6', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML7, scale, 'ML_7', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML8_hyperechoic, scale, 'ML_8_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML9_hyperechoic, scale, 'ML_9_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML10_hyperechoic, scale, 'ML_10_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_ML11_hyperechoic, scale, 'ML_11_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_MV_hyperechoic, scale, 'MV_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_MV_hypoechoic, scale, 'MV_hypoechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_DAS_Alpinion_hyperechoic, scale, 'DAS_hyperechoic', dynamic_range_alpinion, diff);
%plot_alpinion_image(b_data_DAS_Alpinion_hypoechoic, scale, 'DAS_hypoechoic', dynamic_range_alpinion, diff);

% diff images

plot_alpinion_image(b_data_ML4, scale, 'diff_ML_4', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML5, scale, 'diff_ML_5', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML6, scale, 'diff_ML_6', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML8_hyperechoic, scale, 'diff_ML_8_hyperechoic', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML9_hyperechoic, scale, 'diff_ML_9_hyperechoic', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML10_hyperechoic, scale, 'diff_ML_10_hyperechoic', dynamic_range_alpinion, b_data_MV_hyperechoic);
plot_alpinion_image(b_data_ML11_hyperechoic, scale, 'diff_ML_11_hyperechoic', dynamic_range_alpinion, b_data_MV_hyperechoic);

plot_alpinion_image(b_data_ML7, scale, 'diff_ML_7', dynamic_range_alpinion, b_data_MV_hypoechoic);

%% Verasonics Plots
dynamic_range_verasonics = [-70 0];
extract_frame = true;
first_mv_frame = b_data_MV_Verasonics.data(:,:,:,1);
scale = max(first_mv_frame(:));
diff = false;

%plot_verasonics_image(b_data_ML8_Verasonics, scale, 'ML_8_Verasonics', dynamic_range_verasonics, extract_frame, diff);
%plot_verasonics_image(b_data_ML9_Verasonics, scale, 'ML_9_Verasonics', dynamic_range_verasonics, extract_frame, diff);
%plot_verasonics_image(b_data_ML10_Verasonics, scale, 'ML_10_Verasonics', dynamic_range_verasonics, extract_frame, diff);
%plot_verasonics_image(b_data_ML11_Verasonics, scale, 'ML_11_Verasonics', dynamic_range_verasonics, false, diff);
%plot_verasonics_image(b_data_MV_Verasonics, scale, 'MV_Verasonics', dynamic_range_verasonics, extract_frame, diff);
%plot_verasonics_image(b_data_DAS_Verasonics, scale, 'DAS_Verasonics', dynamic_range_verasonics, extract_frame, diff);

% diff images
%dynamic_range_verasonics = [-70 0];
plot_verasonics_image(b_data_ML8_Verasonics, scale, 'diff_ML_8_Verasonics', dynamic_range_verasonics, extract_frame, b_data_MV_Verasonics);
plot_verasonics_image(b_data_ML9_Verasonics, scale, 'diff_ML_9_Verasonics', dynamic_range_verasonics, extract_frame, b_data_MV_Verasonics);
plot_verasonics_image(b_data_ML10_Verasonics, scale, 'diff_ML_10_Verasonics', dynamic_range_verasonics, extract_frame, b_data_MV_Verasonics);
plot_verasonics_image(b_data_ML11_Verasonics, scale, 'diff_ML_11_Verasonics', dynamic_range_verasonics, false, b_data_MV_Verasonics);

%% Appodizaation plots 
L = 64;
ham = hamming(L);
han = hann(L);
black = blackman(L);

w = 2;

hold on;
plot(x, ham, '-', 'LineWidth',w)
plot(x, han, '--', 'LineWidth',w)
plot(x, black, ':', 'LineWidth',w)
legend('Hamming', 'Hanning', 'Blackman')
xlabel('Array element')
ylabel('Apodization weight')
set(gca,'XLim',[0 L],'YLim',[0 1],'fontsize',15)
saveas(gcf, 'images\Apodization_windows.svg')
hold off;

%%
function plot_alpinion_image(b_data, scale, name, dynamic_range, diff)
    font = 15;
    pos = [1000 918 560-180 420];
    
    img = reshape(b_data.data, [512,512]);
    x = b_data.scan.x_axis*1000;
    z = b_data.scan.z_axis*1000;
    
    if diff == false
        % Do nothing
    else
        other_img = reshape(diff.data, [512,512]);
        img = img - other_img;
    end
    
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

function plot_verasonics_image(b_data, scale, name, dynamic_range, extract_frame, diff)
    font = 15;
    pos = [1000 918 560-100 420];

    if extract_frame
        data = b_data.data(:,:,:,1); % Get first image
    else
        data = b_data.data;
    end
    
    scan = b_data.scan;
    x_matrix=reshape(scan.x,[scan.N_depth_axis scan.N_azimuth_axis]);
    x = x_matrix*1e3;
    z_matrix=reshape(scan.z,[scan.N_depth_axis scan.N_azimuth_axis]);
    z = -z_matrix*1e3;
    
    image = abs(reshape(data,scan.N_depth_axis,scan.N_azimuth_axis)./scale);
    
    if diff == false
        % Do nothing
    else
        other_data = diff.data(:,:,:,1); % Get first image
        other_img = abs(reshape(other_data,scan.N_depth_axis,scan.N_azimuth_axis)./scale);
        size(other_img)
        image = image - other_img;
    end

    figure;
    pcolor(x,z,db(image));
    shading(gca,'flat');
    ylabel(['z [mm]']); xlabel('x [mm]');
    colormap gray; 
    caxis(gca, dynamic_range); 
    hcb = colorbar;
    hcb.Title.String = 'dB';
    set(gca,'fontsize',font);
    set(gcf, 'position',pos);
    saveas(gcf, strcat('images\', name, '.svg'))
end
