function fim = extractRandomFeatures(pim,kln, kc, ct, bw, bs)
% 
% Processes a given image using the algorithm explained in the paper.
% kln,  is the weighting kernel that will be used in local neighborhoods
% kc , convolution kernels kc.layer1, kc.layer2
% ct , connection table for convolutions, ct.layer1, ct.layer2
% bw , width of boxcar kernel for average down sampling bw.layer1
%      bw.layer2
% bs , step size for average down sampling, bs.layer1, bs.layer2
% 
fpim = zeros([1 size(pim)]);
fpim(1,:) = pim(:);
f1 = extractFeatureLayer(fpim, kc.layer1 ,ct.layer1 ,kln, bw.layer1 ,bs.layer1);
fim = extractFeatureLayer(f1, kc.layer2 ,ct.layer2 ,kln, bw.layer2 ,bs.layer2);

function dsim = extractFeatureLayer(pim,kc,ct,kln,bw,bs)
%
% Does one layer of feature extraction.
% pim, input feature maps
% kc , convolution kernels
% ct , connection table for convolutions
% kln, kernel for local normalization
% bw , width of boxcar kernel for average down sampling
% bs , step size for average down sampling
%

%
% 1. convolve input features with kernels
% 
cim = convolve(pim,kc,ct);
%
% 2. rectify with absolute val
%
rim = abs(cim);
%
% 3. local normalization
%
lnim = localnorm(rim,kln);
%
% 4. average down sampling
%
dsim = avdown(lnim,bw,bs);

function out = avdown(in,bw,bs)
%
% Do average down sampling of input with a boxcar filter of width bw and
% downsampling step size of bs
%
ker = zeros(bw);
ker(:) = 1/(bw*bw);
oi = (size(in,2)-bw)/bs + 1;
oj = (size(in,3)-bw)/bs + 1;
out = zeros(size(in,1),oi,oj);
for i=1:size(in,1)
    from = squeeze(in(i,:,:));
    to = conv2(from,ker,'valid');
    dsi = 1:bs:size(to,1);
    dsj = 1:bs:size(to,2);
    out(i,:,:) = to(dsi,dsj);
end


function out = convolve(in,ck,ct)
%
% Convolutions of input features (in) with kernels (ck) according
% to connection table (ct)
%
noutf = max(squeeze(ct(:,2)));
nouti = size(in,2)-size(ck,2)+1;
noutj = size(in,3)-size(ck,3)+1;
out = zeros(noutf, nouti, noutj);

for icon = 1:size(ct,1)
    ker = squeeze(ck(icon,:,:));
    from = squeeze(in(ct(icon,1),:,:));
    to = conv2(from,ker,'valid');
    out(ct(icon,2),:,:) = squeeze(out(ct(icon,2),:,:)) + to(:,:);
end

    
function lnim = localnorm(in,k)
%
% given a set of feature maps, performs local normalization
% out = (in -mean(in))/std(in)
% mean and std are defined over local neighborhoods that span
% all feature maps and a local spatial neighborhood
%

% steup the multi dim kernel
% so now we have a 3d kernel, gaussion along dims 2 and 3
% and just averaging along dim 1 wnich is also weighted by the
% gaussian in other dims
ker = zeros(size(in,1),size(k,1),size(k,2));
for i=1:size(in,1)
    ker(i,:) = k(:);
end
ker = ker / sum(ker(:));

% mu = E(in)
inmean = multiconv(in,ker);
% in-mu
inzmean = in - inmean;
% (in - mu)^2
inzmeansq = inzmean .* inzmean;
% std = sqrt ( E (in - mu)^2 )
instd = sqrt(multiconv(inzmeansq,ker));
% threshold std with the mean
mstd = mean(instd(:));
instd(instd<mstd) = mstd;
% scale the input by (in - mu) / std
lnim = inzmean ./ instd;

function out = multiconv(in,ker)
%
% this is basically 3d convolution without zero padding in 3rd
% dimension
%
out = zeros(size(in));
for i=1:size(in,1)
    cin = squeeze(in(i,:,:));
    cker = squeeze(ker(i,:,:));
    out(i,:,:) = conv2(cin,cker,'same');
end
sout = squeeze(sum(out));
for i=1:size(out,1)
    out(i,:) = sout(:);
end
