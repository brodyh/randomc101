
addpath('/home/koray/dev/software/matlab');

maxNumCompThreads(1);

params.eta = 1e-3;
params.nepoch = 30;
params.decay = 0.99;
params.outdir = '../outputs';

trdata.data = readBinLushMatrix('../data/features/train-features-forget-set0.mat');
trdata.labels = readBinLushMatrix('../data/features/train-labl0_may1308.mat')';

tedata.data = readBinLushMatrix('../data/features/test-features-forget-set0.mat');
tedata.labels = readBinLushMatrix('../data/features/test-labl0_may1308.mat')';

onlinegd(trdata, tedata, params);

