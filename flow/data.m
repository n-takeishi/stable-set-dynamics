clearvars;
seed=1234;
rng(seed);
load('ibpm_result.mat', 'VORTALL');
dt=0.2;

% pca
[coeff, score, latent] = pca(VORTALL.');
idx = find(cumsum(latent)/sum(latent)>1-1e-3, 1, 'first');

coeff = coeff(:,1:idx);
score = score(:,1:idx);
latent = latent(1:idx);

% normalization
normalizer_Y = max(abs(score(:,1)));
score = score / normalizer_Y;
normalizer_dotY = 0.1;

% tr = 301:700;
% va = 701:800;
% te = 801:1300;
tr = 1:600;
va = 601:800;
te = 801:1800;
in = tr(1);

%%

noisesigma = 5e-3;

% save data
T = reshape((tr-in)*dt, 1, length(tr));
YY = reshape(score(tr(1):tr(end)+1,:), 1, length(tr)+1, idx);
YY = YY + randn(size(YY))*noisesigma;
Y = YY(:,1:end-1,:); Y_ = YY(:,2:end,:);
dotY = (Y_ - Y)/dt/normalizer_dotY;
save('data_tr.mat', 'T', 'Y', 'dotY');

T = reshape((va-in)*dt, 1, length(va));
YY = reshape(score(va(1):va(end)+1,:), 1, length(va)+1, idx);
YY = YY + randn(size(YY))*noisesigma;
Y = YY(:,1:end-1,:); Y_ = YY(:,2:end,:);
dotY = (Y_ - Y)/dt/normalizer_dotY;
save('data_va.mat', 'T', 'Y', 'dotY');

T = reshape((te-in)*dt, 1, length(te));
YY = reshape(score(te(1):te(end)+1,:), 1, length(te)+1, idx);
YY = YY + randn(size(YY))*noisesigma;
Y = YY(:,1:end-1,:); Y_ = YY(:,2:end,:);
dotY = (Y_ - Y)/dt/normalizer_dotY;
save('data_te.mat', 'T', 'Y', 'dotY');

%% save info
clearvars VORTALL score;
clearvars T YY Y Y_ dotY in;
save('data_info.mat');
