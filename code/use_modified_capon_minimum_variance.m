function b_data_MV = use_modified_capon_minimum_variance(name, scan_type, call_type, Ria_type)
    % data location
    url='http://ustb.no/datasets/';      % if not found downloaded from here
    data_path = 'C:\Users\espen\Documents\Skole\MASTER\code\data\';
    local_path = strcat(data_path, name, '\'); % location of example data
    addpath(local_path);
    
    % check if the file is available in the local path or downloads otherwise
    tools.download(strcat(name, '.uff'), url, local_path);
    channel_data = uff.read_object([local_path, strcat(name, '.uff')],'/channel_data');
    
    % Define the scan
    if strcmp(scan_type, 'Alpinion') %scan_type == 'Alpinion'
        scan = uff.linear_scan();
        scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512).';
        scan.z_axis = linspace(1e-3,50e-3,512).';
        
    elseif strcmp(scan_type, 'Verasonics') %scan_type == 'Verasonics'
        depth_axis = linspace(0e-3,110e-3,1024).';
        azimuth_axis = zeros(channel_data.N_waves,1);
        for n = 1:channel_data.N_waves
            azimuth_axis(n) = channel_data.sequence(n).source.azimuth;
        end
        scan = uff.sector_scan('azimuth_axis',azimuth_axis,'depth_axis',depth_axis);
        
    elseif strcmp(scan_type, 'PICMUS') %scan_type == 'PICMUS'
        scan = uff.linear_scan();
        scan.x_axis = linspace(channel_data.probe.x(1),channel_data.probe.x(end),512)';
        scan.z_axis = linspace(5e-3,50e-3,512)';
    end
    
    % Transmit beamforming (before MV)
    mid = midprocess.das();
    mid.channel_data = channel_data;
    mid.scan = scan;
    mid.dimension = dimension.transmit();
    mid.transmit_apodization.window = uff.window.none;
    mid.receive_apodization.window = uff.window.none;
    b_data_transmit = mid.go();
    
    % MV
    post = modified_capon_minimum_variance();
    post.channel_data = channel_data;
    post.input = b_data_transmit;
    post.dimension = dimension.receive();
    post.scan = scan;
    %post.transmit_apodization.window = mid.receive_apodization;
    %post.receive_apodization.window = mid.transmit_apodization;

    post.L_elements = 16; % subarray size
    post.K_in_lambda = 1; % temporal averaging factor
    post.regCoef = 0; % regularization factor

    b_data_MV = post.go(name, call_type, Ria_type);
end
