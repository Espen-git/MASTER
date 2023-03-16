%%
clear all
close all
run('generate_data.m')
filetype = '-depsc'; % eps with colour

%% R

all_R = zeros(P,M,M);
for index = 1:P

    x = squeeze(all_x(index,:,:));
    R = x*x' / N;
    all_R(index,:,:) = R;
    
    %R_max = max(abs(R),[],'all');
    %figure()
    %imagesc(abs(R)/R_max)
    %a = colorbar;
    %a.Label.String = 'Inter-element correlation';
    %xlabel('Elment number')
    %ylabel('Elment number')
    %title('Normalized spacial correlation matrix of R')

end

%% Time R inversion
tic;
for index = 1:P
    Rinv = inv(squeeze(all_R(index,:,:)));
end
toc;
% Elapsed time is 0.000100 seconds.


%% Test R*R^-1 = I
I_h = R * Rinv;
I = eye(10,'like',I_h);

figure;
imagesc(abs(I_h));
colorbar;

figure;
imagesc(abs(I));
colorbar;

figure;
imagesc(abs(I-I_h));
colorbar;

sum(sum(abs(I-I_h)))
%%

save('data.mat','all_R','all_x');


