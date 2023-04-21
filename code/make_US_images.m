%%
name = 'Alpinion_L3-8_CPWC_hyperechoic_scatterers';
image_dir = strcat('C:\Users\espen\Documents\Skole\MASTER\code\data\', name);
Ria_type = 'Ria_Test4';
Ria_type = 'Ria';
file_path = strcat(image_dir,'\',Ria_type,'\');
path = strcat(file_path,num2str(1),'_',num2str(1),'_',num2str(1),'.mat');
%Ria = cell2mat(struct2cell(load(path)));
Ria = load(path);
%%
Ria.Ria

%%
b_data_ML = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 2, 'Ria_Test4');
%%
b_data_Real = use_modified_capon_minimum_variance('Alpinion_L3-8_CPWC_hyperechoic_scatterers', 'Alpinion', 0, 'Ria');

%% Plot
b_data_ML.plot([],['ML'],[80],[],[],[],[],'dark');      % Display
b_data_Real.plot([],['Real'],[80],[],[],[],[],'dark');      % Display