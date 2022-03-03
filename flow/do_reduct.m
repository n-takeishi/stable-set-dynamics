clearvars;

ny = 199;  % number of grid points in y-direction
nx = 449;  % number of grid points in x-direction
indir = './result/';
outdir = './reduct/';
frames = [1, 100:100:1000];

load('data_info.mat');
seed=1;

mkdir(outdir);

%% truth

load('data_te.mat');
Y_truth = reshape(Y,numel(te),numel(latent))*normalizer_Y;
Y_truth = Y_truth * coeff.';

% save images
for frame=frames
    f = plotcyl(reshape(Y_truth(frame,:),ny,nx));
    saveas(f, sprintf('%s/truth_%05d.png', outdir, frame));
    close(f);
end

%% vanilla

Y1 = importdata([indir,'/vanilla_test_pred_epiidx0.txt']);
Y1 = Y1 * normalizer_Y;
Y1 = Y1 * coeff.';

for frame=frames
    f = plotcyl(reshape(Y1(frame,:),ny,nx));
    saveas(f, sprintf('%s/vanilla_%05d.png', outdir, frame));
    close(f);
end


%% staeq

Y2 = importdata([indir,'/staeq_test_pred_epiidx0.txt']);
Y2 = Y2 * normalizer_Y;
Y2 = Y2 * coeff.';

for frame=frames
    f = plotcyl(reshape(Y2(frame,:),ny,nx));
    saveas(f, sprintf('%s/staeq_%05d.png', outdir, frame));
    close(f);
end
    

%% stainv

Y3 = importdata([indir,'/stainv_test_pred_epiidx0.txt']);
Y3 = Y3 * normalizer_Y;
Y3 = Y3 * coeff.';

for frame=frames
    f = plotcyl(reshape(Y3(frame,:),ny,nx));
    saveas(f, sprintf('%s/stainv_%05d.png', outdir, frame));
    close(f);
end
