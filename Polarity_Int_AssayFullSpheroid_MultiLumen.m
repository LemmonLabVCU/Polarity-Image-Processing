%% Polarity Measurements of Spheroid Cross-Sectional Area

% Cell-Matrix Mechanobiology Lab | VCU Department of Biomedical Engineering
% Contact: Kristin Kim; kimkp@vcu.edu
% Date Modified: 2-23-2024

% Goals of the code: 
% The goal of this code is to use images of the different channels
% taken from IF imaging to define the intensity of channels at the
% different polar ends of the spheroid. Expect that spheroids are divided
% into an apical and basolateral region, and that proteins defined in
% one or more channels are polar proteins/ expressed on one side more than
% the other. To improve calculations, the DAPI channel is used to divide 
% cells/ spheroid, as well as the composite image to define a mask of the
% total spheroid for intensity calculations to remove possible background
% noise in the raw image files.

% Another goal is to improve on the automated measurements to improve data
% analysis.

%% Step 1: Clear MATLAB command window, workspace and close any popup windows open: 
clear;
close all;

%% Step 2: Call in image files by adding their folder to MATLAB path

% identify the folder being called in
defname = fullfile(cd, '*.*');
		[name, folder] = uigetfile(defname, 'Select an image file');
		if name == 0
			% User clicked the Cancel button.
			return;
        end
        
% Get the full filename, with path prepended.
fullname = fullfile(folder, name);

% Check if file exists.
if ~exist(fullname, 'file')
	% File doesn't exist -- didn't find it there.  Check the search path for it.
	fullname = name; % No path this time.
    
	if ~exist(fullname, 'file')
		% Still didn't find it. Alert user via error code.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullname);
		uiwait(warndlg(errorMessage));
		return;
        
    end
    
end

% Call in meta data for images stored in variable [mainFolder]:
mfn = mfilename;
Folder = folder; addpath(Folder);
mainFolder = dir(Folder); mainFolder(1:2) = []; mainFolder(end) = [];

ImParam.Folder = Folder; % save the name of the folder being run through
%% Step 3: Making identifiers for the different channels used
% Warning: Before Running the Code, ensure that the name of the image for
% every channel is the same.

% For example, the group of exported images should look like the following, 
% where [Image_01] is the 'name' of the image file, adn '_t0c0.tif' is the 
% ending identifier of the channel. 
%   Image_01_t0c0.tif
%   Image_01_t0c1.tif
%   Image_01_t0c2.tif
%   Image_01_t0c3.tif
%   Image_01_t0c0-3.tif

idx  = [];               % the number of distinct images in the folder
ImParam.ImTag = name(end-2:end); % Define image file type (jpg, tif, etc)

% Special Channel Considerations
ImParam.CompIdx = 0;    % if there is a composite channel
ImParam.nucIdx = 0;     % if there is a nuclei channel
ImParam.PolarIdx = [];  % separate polarity marker channels
ImParam.delIdx = [];    % Initialize a vector that saves images that aren't run correctly

% Getting rid of non-image files
for i = 1:length(mainFolder) % removing non-image files like thumbs.dx
    clength = length(mainFolder(i).name);
    if clength < 6  
        idx = [idx, i];
        
    elseif mainFolder(i).name(end - 2:end) ~= ImParam.ImTag % another step to removing non-image files like .czi or .lifs
        idx = [idx, i];        
    end    
end

mainFolder(idx) = []; % remove all non-image files from the mainFolder variable

%% Step 4: Define your channels for easier data organization later on

button = menu('How many channels do you have?', '2', '3', '4', '5', 'Exit');
switch button
	case 1
		ImParam.numCh = 2;
	case 2
        ImParam.numCh = 3;
	case 3
        ImParam.numCh = 4;
    case 4
        ImParam.numCh = 5;
    case 5
        return;
end

Tag_lngth = 0;
Ch_Tag = {}; % Initialize Channel Tag cell
EndV = []; % length of channel tag

for i = 1:ImParam.numCh  % Input names of each channel based on what pops up in the figure
    fig = figure(1); subplot(2,3,i); 
    Ch = imread(mainFolder(i).name);
    imshow(Ch);
    
    button = menu('Label channel names.', 'Composite', 'DAPI', 'APICAL', 'Other', 'Exit');
    switch button
	    case 1
		    ImParam.chNames{i} = 'Composite';
            ImParam.CompIdx = i;
	    case 2
            ImParam.chNames{i} = 'DAPI';
            ImParam.nucIdx = i;
        case 3
            ImParam.chNames{i} = 'APICAL';
            ImParam.apicalIdx = i;
	    case 4
            prompt = {'Label channel names:'};
            dlgtitle = 'Channel name: ';
            ImParam.chNames{i} = inputdlg(prompt, dlgtitle);
            ImParam.PolarIdx = [ImParam.PolarIdx, i];
        case 5
            return;
    end

    if Tag_lngth > 0 % Find where the differnet channel names differ.
    
    elseif length(mainFolder(i).name) ~= length(mainFolder(i+1).name)

    else
        Tag_lngth = find(mainFolder(i).name ~= mainFolder(i+1).name);
    end
end

close(1); % close Figure window

%% Step 5: Defining the end tags for each channel and organizing all images
Prefix = mainFolder(i).name(1:Tag_lngth-2);
for i = 1:ImParam.numCh 
    Ch_Tag{i} = mainFolder(i).name(length(Prefix):end);
    EndV(i) = length(Ch_Tag{i}) - 1;
end

indFinal = {};
for i = 1:ImParam.numCh
    idx = [];
    for j = 1:length(mainFolder)
        if length(mainFolder(j).name(end-EndV(i):end)) > length(Ch_Tag{i})
            
        elseif mainFolder(j).name(end-EndV(i):end) == Ch_Tag{i}
            idx = [idx j];
            
        end
    end    
    indFinal{i} = idx;
    
    for n = 1:length(indFinal{i})
        Ch_Files{n, i} = mainFolder(indFinal{i}(n)).name; % THE IMPORTANT VARIABLE THAT WILL BE CALLED A LOT
        Data.ImgName{n} = Ch_Files{n,1}(1:end-EndV-1); 
            
    end
end

ImParam.numFiles = length(idx);

%% Step 6: Calling in Images using imread function

% call in all channels of the image.
for n = 1:ImParam.numCh
    for i = 1:ImParam.numFiles
        tempFile = imread(Ch_Files{i,n});

        if size(tempFile, 3) == 3    % if the channels are RGB/ XxYxC, convert to grayscale image
            tempFile = rgb2gray(tempFile); 
            Data.ImgFiles{i,n} = tempFile; 
        else
            Data.ImgFiles{i,n} = tempFile;
        end
    end
end

%% Step 7: Organizing Data + Initializing Analysis Variables

[ImParam.W, ImParam.L] = size(Data.ImgFiles{1,1}); % height and width of images

Data.compImg = cell(1,ImParam.numFiles); % binary of combined channels or composite image
Data.binImg  = cell(1,ImParam.numFiles);  % binarized images
Data.SumCh   = cell(1, ImParam.numFiles); % sum of pixels

LumenCnt = 1;

for n = 1:ImParam.numFiles % running analysis through each image

    % Default Parameters 
    adjustThresh = 0.3; % max intensity adjustment
    diskSize = 2;
    removeObjSize = 500; % pixel area
    flag = 0;
    
    stats1 = [];
    Data.compImg{n} = Data.ImgFiles{1,1}.*0; % initialize 0-filled matrix

    % Making a composite binary image
    while flag == 0
               
    if ImParam.CompIdx > 0 % if there is a composite image, use that instead
   
        tempImg = Data.ImgFiles{n,ImParam.CompIdx}; 
        tempImg = imadjust(tempImg, [0, adjustThresh]);
        tempImg = imbinarize(tempImg);
        tempImg = imfill(tempImg, 'holes');
        tempImg = bwareaopen(tempImg, removeObjSize);
        Data.binImg{n, i} = tempImg;
        Data.compImg{n} = tempImg;

    else % add all channel images together
        for i = 1:ImParam.numCh
            tempImg = Data.ImgFiles{n,i};
            tempImg = imbinarize(tempImg);
            tempImg = bwareaopen(tempImg, removeObjSize); % remove anything smaller than 50 pixels
            tempImg = imfill(tempImg, 'holes'); % fill any empty holes in the image
            
            tempImg = bwareaopen(tempImg, removeObjSize);
            Data.compImg{n} = Data.compImg{n} + tempImg;           
        end
    end
    
    % Finding lumen area of the spheroids
    tempImg = Data.ImgFiles{n, ImParam.apicalIdx};
    tempImg = imadjust(tempImg, [0, adjustThresh]); 
    tempImg = imbinarize(tempImg); 
    tempImg = bwareaopen(tempImg, removeObjSize); 
    tempImg = imfill(tempImg, 'holes'); 
    
    se = strel('disk', diskSize); % window for image dilation
    LUMENbin = imdilate(tempImg, se);
    Data.lumenImg{n} = LUMENbin;
    
    % Finding Nuclei using DAPI channel
    tempImg = Data.ImgFiles{n, ImParam.nucIdx};
    tempImg = imbinarize(tempImg);
    tempImg = bwareaopen(tempImg, removeObjSize);
    tempImg = immultiply(tempImg, Data.compImg{n}); % remove random noise outside of spheroid
    tempImg = tempImg - Data.lumenImg{n}; % remove nuclei inside lumen
    Data.nucImg{n} = tempImg;
    
    % plotting binarized image
    figure(2); % initialize another figure
    subplot(3,1,1); imshow(Data.compImg{n}); title([Data.ImgName{n}]); hold on;
    subplot(3,1,2); imshow(Data.lumenImg{n}); title('Lumen'); 
    subplot(3,1,3); imshow(Data.nucImg{n}); title('DAPI'); 
    
        button = menu('Does this look ok?', 'Yes', 'No', 'Exit');
        switch button
            case 1
                flag = 1;
            case 2
                prompt = {'Enter Max Intensity Threshold (must be < 1):', 'Enter dilation disk size (suggest keep below 10):', 'Enter max object size to remove: '};
                dlgtitle = 'Image Preprocessing: ';
                fieldsize = [1 45; 1 45; 1 45];
                definput = {'0.3','2', '500'};
                answer = inputdlg(prompt,dlgtitle,fieldsize,definput);
                adjustThresh = str2double(answer{1}); % max intensity adjustment
                diskSize = str2double(answer{2}); % window for image dilation
                removeObjSize = str2double(answer{3}); % pixel area
                
            case 3
                return;
        end
    end
    
    close(2);
    
    figure(3); 
    
    % Image Segmentation - separating spheroid based on the number of
    % lumens it has [From APICAL Channel].
    
    COMPbin  = Data.compImg{n};
    LUMENbin = Data.lumenImg{n};    
    D = -bwdist(~COMPbin);
    L = watershed(D);
    COMPbin2 = COMPbin;
    COMPbin2(L == 0) = 0;
    mask = LUMENbin;
    
    subplot(3,1,1); 
    imshowpair(COMPbin, mask, 'blend'); title('Binary Output: Cells + LUMEN');

    D2 = imimposemin(D, mask);
    Ld2 = watershed(D2);
    COMPbin3 = COMPbin;
    COMPbin3(Ld2 == 0) = 0;
    
    Ld3 = immultiply(COMPbin, Ld2);
    subplot(3,1,2); 
    imshow(COMPbin); hold on;
    himage = imshow(label2rgb(Ld3)); title('Segmented Area Outputs');
    himage.AlphaData = 0.3;
    
    close(3);
    
    % Find average intensity of each channel going through each elliptical section of the spheroid
    NumLumen = double(max(max(Ld3)));
    
    figure(4);
    subplot(NumLumen + 1, 1, 1); 
    imshow(COMPbin); hold on;
    himage = imshow(label2rgb(Ld3)); title(Data.ImgName{n});
    himage.AlphaData = 0.3;
    
    for dd = 1:NumLumen
        LumenMask = COMPbin.*0;
        Mind = find(Ld3 == dd);
        LumenMask(Mind) = 1;

        subplot(NumLumen + 1, 1, dd + 1); imshow(LumenMask); hold on;

        stats1 = regionprops(LumenMask, 'all');
        c = stats1.Centroid;
        text(c(1), c(2), sprintf('%d', dd), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color','red');

        % find centroid position
        r = round(stats1.MajorAxisLength/2) + 10;
        xc = round(stats1.Centroid(1)); yc = round(stats1.Centroid(2));

    % find centroid position
        r = round(stats1.MajorAxisLength/2) + 10;
        xc = round(stats1.Centroid(1)); yc = round(stats1.Centroid(2));

        % define lines passing through circle that will be used to plot
        % intensity profiles
        theta = linspace(0,2*pi);
        x = round(r*cos(theta) + xc); y = round(r*sin(theta) + yc); % rounded to define whole number positions

        how_many_point = 7; % 360 / 45;  % number of lines defined in polar coordinates
        coarse_theta = linspace(0, 2*pi, how_many_point + 1);

        xs = round(xc + r*cos(coarse_theta)); % x(0) values
        xs2 = round(xc - r*cos(coarse_theta)); % x(end) values

        ys = round(yc + r*sin(coarse_theta));  % y(0) values
        ys2 = round(yc - r*sin(coarse_theta)); % y(end) values

        Data.SumCh{LumenCnt} = zeros(r + 2, ImParam.numCh); % initialized vector for sum of intensity profiles
        Data.AvgCh{LumenCnt} = zeros(r + 2, ImParam.numCh); % initialized vector for average intensity profiles
        Data.nucCh{LumenCnt} = zeros(r + 2, ImParam.numCh); % initialized vector for DAPI channel (if present)

        % plotting lines overlaid on the binarized image

        for cc = 1:ImParam.numCh  % going through each image

            for i = 1:length(xs) % going through each line defined

                m = (yc - ys(i)) / (xc - xs(i));

                if m == Inf     % if the line is vertical
                    xx(1:1:r*2) = xs(i);
                    yy(1:1:r*2) = floor(linspace(ys(i), yc, r*2));
                elseif m == 0   % if the line is horizontal
                    xx(1:1:r*2) = floor(linspace(xs(i), xc, r*2));
                    yy(1:1:r*2) = ys(i);
                else            % if the line is angled
                    xx = floor(linspace(xs(i), xc, r*2));
                    yy = floor( m*(xx - xs(i)) + ys(i) );
                end

                plot(xc, yc, 'r*');
                plot(xx, yy); hold on;

                if ImParam.nucIdx > 0 && cc == ImParam.nucIdx % DAPI = only need positions so we use the binarized image
                    tempImg = Data.nucImg{n};
                    chIntvals = improfile(tempImg, [xs(i) xc], [ys(i), yc]);  % basal: apical

                    Data.SumCh{LumenCnt}(1:length(chIntvals), cc) = Data.SumCh{LumenCnt}(1:length(chIntvals), cc) + chIntvals;
                    Data.AvgCh{LumenCnt}(:, cc) = Data.SumCh{LumenCnt}(:, cc) ./ length(xs);

                else % for all other channels, need to use raw image outputs to get intensity profiles
                    tempImg = Data.ImgFiles{n, cc};
                    chIntvals = improfile(tempImg, [xs(i) xc], [ys(i), yc]);  % basal: apical

                    Data.SumCh{LumenCnt}(1:length(chIntvals), cc) = Data.SumCh{LumenCnt}(1:length(chIntvals), cc) + chIntvals;
                    Data.AvgCh{LumenCnt}(:, cc) = Data.SumCh{LumenCnt}(:, cc) ./ length(xs);
                end

            end

            % Normalizing Distance Measurements to 100 pixels
            X_rng = (1:length(Data.AvgCh{LumenCnt}))./length(Data.AvgCh{LumenCnt})'; 
            X_rng_rnd = round(X_rng, 2);
            Data.X_norm = (0:0.01:1);

            for nn = 1:length(Data.X_norm)
                idxOut = find(X_rng_rnd == Data.X_norm(nn));
                if isempty(idxOut)
                    Data.ChOut{LumenCnt}(nn, cc) = 0;

                else
                    Data.ChOut{LumenCnt}(nn, cc) = mean(Data.AvgCh{LumenCnt}(idxOut, cc));

                end
            end
        end

        % Separating intensity profile into basal v. apical sides:

        for cc = 1:ImParam.numCh

            if ImParam.nucIdx > 0 % if there is a DAPI channel present

                PolarityIdx = find(Data.ChOut{LumenCnt}(:, ImParam.nucIdx) > 0); 
                NUCmid = round(((PolarityIdx(end) - PolarityIdx(1)) / 2) + PolarityIdx(1)); 

                Data.AvgPolarInt{LumenCnt}(1, cc) = mean(Data.ChOut{LumenCnt}(1:NUCmid, cc)); % basal
                Data.AvgPolarInt{LumenCnt}(2, cc) = mean(Data.ChOut{LumenCnt}(NUCmid+1:end, cc)); % apical

                Data.MaxPolarInt{LumenCnt}(1, cc) = max(Data.ChOut{LumenCnt}(1:NUCmid, cc)); % basal
                Data.MaxPolarInt{LumenCnt}(2, cc) = max(Data.ChOut{LumenCnt}(NUCmid+1:end, cc)); % apical

            else % assume midline is at 50 pixels

                Data.AvgPolarInt{LumenCnt}(1, cc) = mean(Data.ChOut{LumenCnt}(1:51, cc)); % basal
                Data.AvgPolarInt{LumenCnt}(2, cc) = mean(Data.ChOut{LumenCnt}(52:end, cc)); % apical

                Data.MaxPolarInt{LumenCnt}(1, cc) = max(Data.ChOut{LumenCnt}(1:51, cc)); % basal
                Data.MaxPolarInt{LumenCnt}(2, cc) = max(Data.ChOut{LumenCnt}(52:end, cc)); % apical

            end

        end

        button = menu('Does this look ok?', 'Yes', 'No', 'Exit');
        switch button
            case 1
                
            case 2
                ImParam.delIdx = [ImParam.delIdx, n];
            case 3
                return;
        end
        
        Data.OutputNames{LumenCnt} = [Data.ImgName{n}, '_LumenCnt', num2str(LumenCnt)];
        LumenCnt = LumenCnt + 1;
    end
end

%% Outputting Data as an excel file:

button = menu('Do you want to export data results?', 'Yes', 'No', 'Exit');
    switch button
	    case 1
            prompt = {'Write file name (Do not include .xlsx at the end to export as an excel file:'};
            dlgtitle = 'File Name: ';
            Xfilename = inputdlg(prompt, dlgtitle);
            Xfilename = append(Xfilename, '.xlsx');
            exportPolarityData(ImParam, Data, string(Xfilename));
	    case 2
            disp(['Run Complete. Number of Images analyzed: ', num2str(ImParam.numFiles)]);
	    case 3
            return;

    end

%%
function exportPolarityData(ImParam, Data, Name)

%   Goal: Export Data collected from run into an excel file

% Name Sheet Names to separate Data Outputs
sheetNames{1} = 'Run Parameters'; % ImParam Outputs
sheetNames{2} = 'Sum Raw Data Output'; % SumCh
sheetNames{3} = 'Avg Raw Data Output'; % AvgCh
sheetNames{4} = 'Normalized Raw Data Output'; % ChOut
sheetNames{5} = 'Max Image Intensity'; % MaxPolarInt
sheetNames{6} = 'Average Image Intensity'; % AvgPolarInt

count = 1;
cLength = [];

for n = 1:length(Data.OutputNames)
    for i = 1:ImParam.numCh
        OutputNames(count) = append(string(Data.OutputNames{n}), ' ', string(ImParam.chNames{i}));
        count = count + 1;
    end
    cLength(n) = length(Data.AvgCh{n}(:,1));
end

[~, idx] = max(cLength);

for n = 1:ImParam.numFiles
    cAdd = cLength(idx) - cLength(n);
    mAdd = NaN([cAdd, ImParam.numCh]);
    
    Data.SumCh{n} = [Data.SumCh{n}; mAdd];
    Data.AvgCh{n} = [Data.AvgCh{n}; mAdd];
end

writetable(struct2table(ImParam,"AsArray", true), Name, 'Sheet', sheetNames{1},'Range', 'A1');

writematrix(cell2mat(Data.SumCh), Name, 'Sheet', sheetNames{2}, 'Range', 'A2');
writematrix(OutputNames, Name, 'Sheet', sheetNames{2}, 'Range', 'A1');

writematrix(cell2mat(Data.AvgCh), Name, 'Sheet', sheetNames{3}, 'Range', 'A2');
writematrix(OutputNames, Name, 'Sheet', sheetNames{3}, 'Range', 'A1');

writematrix([Data.X_norm', cell2mat(Data.ChOut)], Name, 'Sheet', sheetNames{4}, 'Range', 'A2');
writematrix(OutputNames, Name, 'Sheet', sheetNames{4}, 'Range', 'B1');

writematrix(cell2mat(Data.MaxPolarInt), Name, 'Sheet', sheetNames{5}, 'Range', 'B2');
writematrix(OutputNames, Name, 'Sheet', sheetNames{5}, 'Range', 'B1');
writecell({'Basal'; 'Apical'}, Name, 'Sheet', sheetNames{5}, 'Range', 'A2');

writematrix(cell2mat(Data.AvgPolarInt), Name, 'Sheet', sheetNames{6}, 'Range', 'B2');
writematrix(OutputNames, Name, 'Sheet', sheetNames{6}, 'Range', 'B1');
writecell({'Basal'; 'Apical'}, Name, 'Sheet', sheetNames{6}, 'Range', 'A2');

end