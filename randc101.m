
tstr = datestr(now,'yyyymmdd_HHMMSS');

% where to write the outputs
gparams.outdir = '../outputs';
gparams.tstr = tstr;

% Initialize parameters for online gradient descent
gparams.eta = 1e-2;
gparams.nepoch = 30;
gparams.decay = 0.03;

maxNumCompThreads(4);

% can read a previous data instead of processing
% load('../outputs/data.mat');
% first prepare data
[trdata,tedata] = prepareData();
save([gparams.outdir,'/data_',tstr]);

% just run with 1 core
maxNumCompThreads(1);

fprintf('\n\n\n\n%s\n\n\n\n','Training Linear Classifier... (This does not take long)');
% then run classifier
[weights, biases] = onlinegd(trdata, tedata, gparams);
save([gparams.outdir,'/data_',tstr]);