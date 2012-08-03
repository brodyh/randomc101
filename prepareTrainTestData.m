function [trdata, tedata] = prepareTrainTestData(data,labels, ntr, nte)
%  prepares separate training and testing datasets from given
%  complete data
%  data   : a full data matrix [Nimages x Nfeatures x Nfi x Nfj]
%  labels : corrsponding labels for data
%  ntr    : number of training images per class
%  nte    : number of testing images per class
%

nf = size(data,2);
fi = size(data,3);
fj = size(data,4);

nclass = max(labels(:))+1;

trdata.data = zeros(nclass*ntr,nf,fi,fj);
trdata.labels = zeros(nclass*ntr,1);

tedata.data = zeros(nclass*nte,nf,fi,fj);
tedata.labels = zeros(nclass*nte,1);

trcntr = 0;
tecntr = 0;
for iclass = 1:nclass
    % indices that belong to current class
    inds = find(labels==iclass-1);
    nims = length(inds);
    % select randomly ntr for training and nte for test
    % sometimes there is not enough for test, so take whatever is
    % available
    pinds = randperm(nims);
    trinds = inds(pinds(1:ntr));
    teinds = inds(pinds(ntr+1:min(ntr+nte,nims)));
    % collect data
    % training
    trdata.data(trcntr+1:trcntr+length(trinds),:) = data(trinds,:);
    trdata.labels(trcntr+1:trcntr+length(trinds)) = labels(trinds);
    trcntr = trcntr + length(trinds);
    % testing 
    tedata.data(tecntr+1:tecntr+length(teinds),:) = data(teinds,:);
    tedata.labels(tecntr+1:tecntr+length(teinds)) = labels(teinds);
    tecntr = tecntr + length(teinds);
end

% since testing images might be less than desired, resize
tedata.data = tedata.data(1:tecntr,:,:,:);
tedata.labels = tedata.labels(1:tecntr);
