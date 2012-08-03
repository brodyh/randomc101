function [w,b] = onlinegd(trdata, tedata, params)
%
% Does training using online stochastic gradient based methods
%
%  Input variables
%  trdata          : struct that contains training dataset and labels
%    .data         : training data (each row is a sample)
%    .labels       : training labels
%  tedata          : testing dataset and labels
%    .data         : testing data (each row is a sample)
%    .labels       : testing labels
%  params          : parameters related to learning
%    .eta          : learning rate
%    .decay        : learning rate decay at each epoch
%    .outdir       : output directory to write results
%

%% get data from structs
dtr = trdata.data;
ltr = trdata.labels;
dte = tedata.data;
lte = tedata.labels;
ntrain = size(dtr,1);
ntest = size(dte,1);

nclass = max([ltr(:) ; lte(:)])+1;

%% permute training data
trind = randperm(ntrain);

%% training and test data can be one sample per row.
dtr = dtr(trind,:);
ltr = ltr(trind,:);
dte = dte(1:ntest,:);
lte = lte(1:ntest,:);

%% number of epochs over the training set
nepoch = params.nepoch;
oeta = params.eta;
ceta = params.eta;

decay = params.decay;

%% init loss and error rates
trloss = zeros(nepoch,1);
tracc = zeros(nepoch,1);
teloss = zeros(nepoch,1);
teacc = zeros(nepoch,1);

%% init parameters
w = -0.015 + 0.03 * rand(nclass,size(dtr,2));
b = zeros(nclass,1);

%% start training
for epoch = 1:nepoch
    %% train
    trloss(epoch) = 0.0;
    tracc(epoch) = 0.0;
    confmat = zeros(nclass);
    fprintf('@@@ EPOCH %d @@@ training : ', epoch);drawnow;
    for iter=1:ntrain
        % take current sample
        x = double(dtr(iter,:)');
        tclass = ltr(iter)+1;
        trgt = zeros(nclass,1);
        trgt(tclass) = 1;
        % run through regressor
        [y,yw,dw,db] = logreg(x,trgt,w,b);
        % get classification
        [mm,pclass] = max(yw);
        confmat(tclass,pclass) = confmat(tclass,pclass)+1;
        % update weights
        w = w - ceta*dw;
        b = b - ceta*db;
        trloss(epoch) = trloss(epoch) + y;
%         if mod(iter,300) == 0
%             fprintf('.');drawnow;
%         end
    end
    trloss(epoch) = trloss(epoch) / ntrain;
    tracc(epoch) = sum( diag(confmat) ./ sum(confmat,2) ) / double(nclass) * 100;
    fprintf('loss = %g, acc = %g\n',trloss(epoch),tracc(epoch));drawnow;
    %% test
    teloss(epoch) = 0.0;
    teacc(epoch) = 0.0;
    confmat = zeros(nclass);
    fprintf('@@@ EPOCH %d @@@ testing : ', epoch);drawnow;
    for iter=1:ntest
        % take current sample
        x = double(dte(iter,:)');
        tclass = lte(iter)+1;
        trgt = zeros(nclass,1);
        trgt(tclass) = 1;
        % run through regressor
        [y,yw] = logreg(x,trgt,w,b,1);
        % get classification
        [mm,pclass] = max(yw);
        confmat(tclass,pclass) = confmat(tclass,pclass)+1;
        teloss(epoch) = teloss(epoch) + y;
%         if mod(iter,300) == 0
%             fprintf('.');drawnow;
%         end
    end
    teloss(epoch) = teloss(epoch) / ntest;
    teacc(epoch) = sum( diag(confmat) ./ sum(confmat,2) ) / double(nclass) * 100;
    fprintf('loss = %g, acc = %g\n',teloss(epoch),teacc(epoch));drawnow;
    % save training info
    save([params.outdir,'/onlinegd_',params.tstr],'trloss','teloss','tracc','teacc','confmat','epoch','w','b','ceta');
    % update learning rate with decay
    ceta = oeta / (1+ epoch*decay);
    fprintf('updated learning rate %g\n',ceta);drawnow;
end
