%%
name = 'Alpinion_L3-8_CPWC_hyperechoic_scatterers';
scan_type = 'Alpinion';
Ria_type = 'True_Ria';
[b_data_DAS, b_data_MV] = use_R_data(name, scan_type, Ria_type);

%b_data_mv.plot([],['MV'],[80],[],[],[],[],'dark');      % Display
%b_data_DAS.plot([],['DAS'],[80],[],[],[],[],'dark');      % Display