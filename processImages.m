function [trdata,trlabels,tedata,telabels] = processImages(imdir,params,ntr,nte)
% imdir is supposed to be 101ObjectCategories directory
% that contains the Caltech 101 data in one folder per
% category
% a resized graysacle version of each image
% will be written

categories = dir(imdir);
cats = {};

% these numbers are fixed for this demo
nf = 256;
fi = 4;
fj = 4;

% we shrink images to 151x151
sz = 151;
params.sz = sz;



if params.c101
    images = struct([]);
    catcntr = 1;
    for i=1:length(categories)
        cat = categories(i);
        if ~strcmp(cat.name,'.') && ~strcmp(cat.name,'..')
            cats(catcntr).name = cat.name;
            ims = dir([imdir,'/',cat.name,'/*.jpg']);
            catind = randperm(length(ims));
            catind = catind(1:min(ntr+nte,length(ims)));
            dn = repmat(struct('dirname',[imdir,'/',cat.name]),length(ims));
            % labels always start from 0 for me
            cn = repmat(struct('class',catcntr-1),length(ims));
            [ims(:).dirname] = dn(:).dirname;
            [ims(:).class] = cn(:).class;
            images = struct([images ; ims(catind)]);
            catcntr = catcntr + 1;
        end
    end
    
    % local norm kernel
    ker = params.ker;
    
    % initialize data and label matrices
    nims = length(images);
    data = zeros(nims,nf,fi,fj);
    labels = zeros(nims,1);
    if params.debug
        len = 1:params.smallSize;
    else
        len = 1:length(images);
    end
    for iim=len
        % get next image
        imname = [images(iim).dirname,'/',images(iim).name];
        [p,basename] = fileparts(imname);
        fprintf('\b\b\b\b\b\b\b\b\b%04d/%04d',iim,length(images));drawnow;
        % read rgb image
        im = imread(imname);
        % convert to grayscale
        if length(size(im)) == 3
            img = rgb2gray(im);
        else
            img = im;
        end
        % resize longer side to sz
        imr = resize_im(img,sz);
        %rimname = [p,'/',basename,'_gray_151.png'];
        %imwrite(imr,rimname);
        % now preprocess as dicarlo
        pim = imPreProcess(imr,ker);
        %pimname = [p,'/',basename,'_prepro.mat'];
        %save(pimname,'pim');
        % then extract random features
        fim = extractRandomFeatures(pim, ker, params.kc, params.ct, params.bw, params.bs);
        %fimname = [p,'/',basename,'_randfeat.mat'];
        %save(fimname,'fim');
        % save this into data and label
        data(iim,:) = fim(:);
        labels(iim) = images(iim).class;
    end
    
    trdata = data;
    trlabels = labels;
    
else
    % init
    trdata = zeros(100,nf, fi, fj);
    trlabels = [];
    tedata = zeros(100,nf, fi, fj);
    telabels = [];
    tecount = 0;
    trcount = 0;
    subSamplePercent = params.subSamplePercent;
    split = params.split;
    %

    imdir = '/Users/Brody/Documents/GradSchool/projects/deepRAEVision/data/2D';
    load([imdir '/../splits/trainTestSplit_' num2str(split) '.mat'],'testingInt');
    instances = dir(imdir);
    fprintf('\b%04d/%04d\n',0,length(instances));drawnow;
    for i=1:length(instances)
        fprintf('\b\b\b\b\b\b\b\b\b\b%04d/%04d\n',i,length(instances));drawnow;
        if ~strcmp(instances(i).name,'.') && ~strcmp(instances(i).name,'..') && ~strcmp(instances(i).name,'.DS_Store')
            instance = load([imdir '/' instances(i).name]);
            instance = instance.allData2D;
            instanceInds = randperm(length(instance));
            instance = instance(instanceInds(1:ceil(subSamplePercent*length(instanceInds))));
            if testingInt(instance(1).classNum) == instance(1).instanceNum
                [tedata tecount] = addInstance(instance, tedata, tecount, params,'te');
                telabels = [telabels; ones(length(instance),1)*instance(1).classNum];
                assert(length(telabels)==tecount);
            else
                [trdata trcount] = addInstance(instance, trdata, trcount, params,'tr');
                trlabels = [trlabels; ones(length(instance),1)*instance(1).classNum];
                assert(length(trlabels) == trcount);
            end
        end
    end
end

trdata = cutData(trdata,trcount);
tedata = cutData(tedata,tecount);
trdata = struct('data',trdata,'labels',trlabels);
tedata = struct('data',tedata,'labels',telabels);




return


function [data count]= addInstance(instance, data, count, params,str)
% local norm kernel
ker = params.ker;
sz = params.sz;
fprintf('%s: %04d/%04d t: %04d',str,0,length(instance), count);
for i = 1:length(instance)
    count = count + 1;
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%s: %04d/%04d t: %04d',str,i,length(instance),count);drawnow;
    if length(size(instance(i).image2D)) == 3
        img = rgb2gray(instance(i).image2D);
    else
        img = instance(i).image2D;
    end
            
    imr = resize_im(img,sz);
    pim = imPreProcess(imr,ker);
    %pimname = [p,'/',basename,'_prepro.mat'];
    %save(pimname,'pim');
    % then extract random features
    fim = extractRandomFeatures(pim, ker, params.kc, params.ct, params.bw, params.bs);
    %fimname = [p,'/',basename,'_randfeat.mat'];
    %save(fimname,'fim');
    % save this into data and label
    if count > size(data,1)
        data(end*2,end,end) = 0;
    end
    data(count,:) = fim(:);
    
end

fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');drawnow;

return


function imres = resize_im(img,sz)
% input has to be grayscale image
% resize the longer side of input image to sz
%

szim = size(img);
[maxs,maxi] = max(szim);
szn = [NaN NaN];

szn(maxi) = sz;
imres = imresize(img,szn);
return;

function b = bsp(n)
b = '';
for i=1:n
    b = [b '\b'];
end
return

function data = cutData(data,length)
data = data(1:length,end,end);
return






