function cdir = getC101(out)
% download caltech 101 data and extract to given folder
% out : should be a folder name, if out does not exist, will be created
%

if (nargin == 0)
    out = '../data/c101';
end

fprintf ('Caltech 101 will be downloaded to %s\n',out);

c101www = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';

if ~isdir(out)
    if mkdir(out) == 0
        error('could not create directory for Caltech 101 download');
    end
end

cdir = [out '/101_ObjectCategories'];

if isdir(cdir)
    fprintf('found already downloaded Caltech 101, skipping download\n');
    return;
end

% download
fprintf ('Warning depending on your internet connection speed, \nThis might take long time');
fprintf('\nUntil the download is finished, the file will **NOT** appear in %s\n',out);
drawnow;
gunzip(c101www,out);
untar([out '/101_ObjectCategories.tar'],out);


