%% Polarity Measurements of Spheroid Cross-Sectional Area

% Cell-Matrix Mechanobiology Lab | VCU Department of Biomedical Engineering
% Contact: Kristin Kim; kimkp@vcu.edu
% Date Modified: 5-30-2025

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
mainFolder = dir(Folder); mainFolder(1:2) = []; %mainFolder(end) = [];

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
ImParam.apicalIdx = 0; % if there is an actin channel
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

prompt = {'Which channels should be used to find lumen mask:'};
    dlgtitle = 'Channel #s: ';
    tempROI_idx = inputdlg(prompt, dlgtitle);
    tempROI_idx = cell2mat(tempROI_idx);
    tempROI_idx = str2num(tempROI_idx);


prompt = {'Which channels do you want to measure colocalization between:'};
    dlgtitle = 'Channel #s: ';
    coloc_idx = inputdlg(prompt, dlgtitle);
    coloc_idx= cell2mat(coloc_idx);
    coloc_idx = str2num(coloc_idx);
    ImParam.colocIdx = coloc_idx;

close(1); % close Figure window


%% Step 5: Defining the end tags for each channel and organizing all images

Prefix = mainFolder(i).name(1:Tag_lngth-2);

for i = 1:ImParam.numCh 
    Ch_Tag{i} = mainFolder(i).name(length(Prefix) + 1:end);
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
        Data.ImgName{n} = Ch_Files{n,1}(1:end-EndV(i)-1); 
            
    end
end

ImParam.numFiles = length(idx);


%% Step 6: Calling in Images using imread function

% call in all channels of the image.
for n = 1:ImParam.numCh

    for i = 1:ImParam.numFiles

        tempFile = [];
        tempFile = imread(Ch_Files{i,n});

        if size(tempFile, 3) == 4    % if the channels are RGB/ XxYxC, convert to grayscale image
            tempFile = rgb2gray(tempFile(:,:,1:3)); 
            Data.ImgFiles{i,n} = tempFile; 

        else 
            Data.ImgFiles{i,n} = im2gray(tempFile);

        end
    end
end

%% Step 7: Organizing Data + Initializing Analysis Variables

[ImParam.W, ImParam.L] = size(Data.ImgFiles{1,1}); % height and width of images

Data.compImg = cell(1,ImParam.numFiles); % binary of combined channels or composite image
Data.binImg  = cell(1,ImParam.numFiles);  % binarized images
Data.SumCh   = cell(1, ImParam.numFiles); % sum of pixels

Data.sumCell = zeros(ImParam.numFiles, ImParam.numCh);
Data.avgCell = zeros(ImParam.numFiles, ImParam.numCh);
Data.sumLumen = zeros(ImParam.numFiles, ImParam.numCh);
Data.avgLumen = zeros(ImParam.numFiles, ImParam.numCh);
Data.sumECM = zeros(ImParam.numFiles, ImParam.numCh);
Data.avgECM = zeros(ImParam.numFiles, ImParam.numCh);

Data.SumBasalCh = zeros(ImParam.numFiles, ImParam.numCh); % initialized vector for sum of intensity profiles
Data.SumApicalCh = zeros(ImParam.numFiles, ImParam.numCh);
Data.AvgBasalCh = zeros(ImParam.numFiles, ImParam.numCh); % initialized vector for average intensity profiles
Data.AvgApicalCh = zeros(ImParam.numFiles, ImParam.numCh);
Data.MaxBasal = zeros(ImParam.numFiles, ImParam.numCh);
Data.MaxApical = zeros(ImParam.numFiles, ImParam.numCh);

Data.CellArea = zeros(ImParam.numFiles,1);
Data.LumenArea = zeros(ImParam.numFiles, 1);
Data.Diameters = zeros(ImParam.numFiles, 2);
Data.LumenDia = zeros(ImParam.numFiles, 2);
Data.colocP = zeros(ImParam.numFiles, 2);

Data.maxCell = zeros(ImParam.numFiles, ImParam.numCh);
Data.maxLumen = zeros(ImParam.numFiles, ImParam.numCh);
Data.maxECM = zeros(ImParam.numFiles, ImParam.numCh);

LumenCnt = [];

%% Looping through each image to do analysis.

prompt = {'Which image do you want to start with?:'};
    dlgtitle = 'n = : ';
    fieldsize = [1 45];
    definput = {'1'};
    tempN = inputdlg(prompt, dlgtitle, fieldsize, definput);
    tempN = cell2mat(tempN);
    tempN = str2num(tempN);

    if isempty(LumenCnt)
        LumenCnt = tempN;
    else
        LumenCnt = LumenCnt - 1;
    end
    

for n = tempN:ImParam.numFiles % running analysis through each image
    
    % Setting up initial parameters for pre-processing images
    diskSize = 2;
    removeObjSize = 2000;

    nucObjSize   = 50;
    nucAdjustmin = 0;
    nucAdjustmax = 0.5;

    SE = strel('disk', 2);
    SE2 = strel('disk', diskSize);
    flag = 0;

    %while flag == 0

        if ImParam.CompIdx > 0 && isempty(tempROI_idx) % use composite image to binarize if present

            tempImg = Data.ImgFiles{n, ImParam.CompIdx};

        else % use chosen channels to make composite image and binarize

            tempImg = Data.ImgFiles{1,1}.*0;

            for i = 1:length(tempROI_idx) 

                if size(Data.ImgFiles{n, tempROI_idx(i)},3) == 4 
                    Data.ImgFiles{n, tempROI_idx(i)} = Data.ImgFiles{n, tempROI_idx(i)}(:,:,3);
                end  

                tempImg = tempImg + Data.ImgFiles{n, tempROI_idx(i)};
            end
        end


        if ImParam.apicalIdx > 0  % If an apical marker is present, use this channel to identify spheroid lumen
            
            tempROI = Data.ImgFiles{n, ImParam.apicalIdx};
            tempROI2 = imbinarize(tempROI);
            tempROI2 = bwareaopen(tempROI2, 100);
            tempROI2 = imdilate(tempROI2, SE2);
            tempROI2 = imclose(tempROI2, SE2); tempROI2 = imfill(tempROI2, 'holes');
            %tempImg  = immultiply(tempROI2, tempImg);

            tempROI = imadjust(tempROI, [0, 0.3]);


        else
            for i = 1:length(tempROI_idx)
                tempROI = tempImg.*0;
                tempROI = tempROI + Data.ImgFiles{n,tempROI_idx(i)};
            end
            tempROI = imadjust(tempROI, [0 0.3]);

        end

        % Creating binarized masks of spheroids
        
        compImg = imadjust(tempImg);
        compBin = imbinarize(compImg); 
        compBin = imfill(compBin, 'holes');
        compBin = bwareaopen(compBin, removeObjSize);
        compBin = imerode(compBin, SE2);
        compBin = imdilate(compBin, SE2);
        
        figure(1); imshow(compImg);  hold on;
        himage = imshow(label2rgb(compBin)); himage.AlphaData = 0.3; hold off;

        button = menu('Does this look ok?', 'Yes', 'No', 'Skip', 'Exit');

            switch button
                case 1

                case 2 % let users fill in missing areas of the spheroid
                    
                    flagC = 0;
                    imshow(compImg); hold on;
                    while flagC == 0
                        cellFillArea = [];

                        cellROI = drawfreehand;
                        cellFillArea = createMask(cellROI);
                        compBin = imbinarize(compBin + cellFillArea);

                        button2 = menu('Add another area?', 'Yes', 'No', 'Exit');  

                        switch button2
                            case 1
                                flagC = 0;
                            case 2                      
                                flagC = 1; 
                            case 3
                                return;
                        end
                    end

                case 3 % skip this image

                    LumenCnt = LumenCnt + 1;
                    continue;

                case 4
                    return;
            end


        % Manual creation of lumen mask:
        numSpheroid = regionprops(compBin, 'Area', 'MajorAxisLength', 'MinorAxisLength');
        lumenMask = zeros(ImParam.W, ImParam.L);
 
        for i = 1:length(numSpheroid)
            flag2 = 0;
            while flag2 == 0
                figure(2); imshow(tempROI); title(['Draw each Lumen: ', num2str(i), '/', num2str(length(numSpheroid))]);
                lumenROI = images.roi.AssistedFreehand(Color="y");
                draw(lumenROI);
                lumenArea = createMask(lumenROI);

                button = menu('Does this look ok?', 'Yes', 'No', 'Exit');
    
                switch button
                    case 1
                        lumenMask = lumenMask + lumenArea;
                        flag2 = 1;
                    case 2
                        
                        flag2 = 0; 
                    case 3
                        return;
                end
            end
        end

        close(2); 

        % Perform watershed to segment image and background, and identify
        % if more than 1 spheroid is present
        figure(1); set(gcf, 'Position', [86, 73, 466, 786]);

        D = -bwdist(~compBin);
        L = watershed(D);
        compBin2 = compBin;
        compBin2(L == 0) = 0;    

        subplot(4,2,4); 
        imshowpair(compBin, lumenMask, 'blend'); title('Binary Output: Cells + LUMEN');
    
        D2 = imimposemin(D, lumenMask);
        Ld2 = watershed(D2);
        compBin3 = compBin;
        compBin3(Ld2 == 0) = 0;
    
        Ld3 = immultiply(compBin, Ld2);
        subplot(4,2,8); 
        imshow(compBin); hold on;
        himage = imshow(label2rgb(Ld3)); title('Segmented Area Outputs');
        himage.AlphaData = 0.3;
    
        s = regionprops(Ld3, 'Centroid');
        for k = 1:numel(s)
            c = s(k).Centroid;
            text(c(1), c(2), sprintf('%d', k), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color','magenta');
        end

        if ImParam.nucIdx > 0
            nucImg = Data.ImgFiles{n, ImParam.nucIdx};
            nucImg = wiener2(nucImg, [5,5]); % remove afterwards
            nucImg = imadjust(nucImg, [nucAdjustmin, nucAdjustmax]);
            nucBin = imbinarize(nucImg);
            nucBin = immultiply(nucBin, compBin);
            nucBin = immultiply(nucBin, ~lumenMask);
            nucBin = bwareaopen(nucBin, nucObjSize);
            nucBin = immultiply(nucBin, Ld3);

            subplot(4,2,3); imshow(label2rgb(nucBin, 'copper')); title('DAPI Bin');

        end

        % Create masks for cell, lumen, and ECM for local intensity analysis
        cellMask = compBin; cellMask = imdilate(cellMask, SE);
        ecmMask = ~cellMask;

        cellMask = cellMask - lumenMask;
        cellMask = immultiply(cellMask, double(Ld3));

        lumenMask = immultiply(lumenMask, double(Ld3));

        subplot(4,2,1); imshow(compImg); title(['Composite Image #: ', num2str(n), '/', num2str(ImParam.numFiles)]);
        subplot(4,2,2); imshow(compBin); title('Composite Bin');

        subplot(4,2,5); imshow(compBin); hold on; nImage = imshow(label2rgb(lumenMask)); title('Lumen Mask');   
        nImage.AlphaData = 0.3;
        
        subplot(4,2,6); imshow(compBin); hold on; cImage = imshow(label2rgb(cellMask)); title('Spheroid Mask');
        cImage.AlphaData = 0.3;
        
        subplot(4,2,7); imshow(ecmMask); title('ECM Mask');

    % If there are spheroids on the edge of the image, or do not match
    % desired morphology, remove before analysis.
    prompt = {'Which spheroids do you want to remove? (Leave a space between each number'};
    dlgtitle = 'Spheroid Number: ';
    Cell_Cnt = inputdlg(prompt, dlgtitle);
    Cell_Cnt = cell2mat(Cell_Cnt);
    Cell_Cnt = str2num(Cell_Cnt);

    for k = 1:length(Cell_Cnt)
        Mind = find(Ld3 == Cell_Cnt(k));
        Ld3(Mind) = 0;
        lumenMask(Mind) = 0;
        cellMask(Mind) = 0;
        ecmMask(Mind) = 0;
    end

    close(1);

    %%
    % Finding average intensity of each channel going through
    % each elliptical section of the spheroid

    lumenInd = unique(Ld3); lumenInd = lumenInd(2:end);
    NumLumen = length(lumenInd);

    for dd = 1:NumLumen % If there are multiple spheroids in the image

        figure(4); set(gcf, 'Position', [156,361,788,520]);

        CellMask1 = compBin.*0;
        LumenMask1 = compBin.*0;

        Mind = find(cellMask == lumenInd(dd));
        Nind = find(lumenMask == lumenInd(dd));

        CellMask1(Mind)  = 1;
        LumenMask1(Nind) = 1;

        if ImParam.nucIdx > 0  % If DAPI channel is present, use to find midpoint of spheroid width
            nucMask1   = compBin.*0;
            Dind = find(nucBin  == lumenInd(dd));
            nucMask1(Dind)   = 1;
            bw = nucMask1;

            % Adjust this based on the gap size you want to fill:
            radius = round(numSpheroid(dd).MajorAxisLength / 5);
            % Pad the edges first to avoid edge effects:
            bwPad = padarray(bw, [radius radius], 0, 'both');
            % Apply the close and skeleton operations:
            bwSkel = bwmorph(imclose(bwPad, strel('disk', radius)), 'skel', Inf);
            % Remove the edge padding:
            bwSkel = bwSkel((1+radius):(end-radius), (1+radius):(end-radius));
            bwSkel = bwmorph(bwSkel, 'spur',Inf);
            % Combine the original and skeleton images:
            nucMask2 = bw | imdilate(bwSkel, strel('disk', 2));
            
            subplot(1,2,1); imshow(nucMask2);

            button = menu('Does this look ok?', 'Yes', 'No', 'Exit');
            switch button
                case 1
                   nucMask1 = nucMask2;
                case 2
                  
                  lumenROI = drawfreehand;
                  
                  button2 = menu('Happy with this?', 'Yes', 'No', 'Exit');

                    switch button2
                        case 1
                          tempLine = createMask(lumenROI);
                          tempLine = edge(tempLine);
                          tempLine = imdilate(tempLine, strel('disk', 2));
                          nucMask1 = imbinarize(nucMask1 + tempLine);
                        case 2

                          lumenROI = drawfreehand;
                          tempLine = createMask(lumenROI);
                          tempLine = edge(tempLine);
                          tempLine = imdilate(tempLine, strel('disk', 2));
                          nucMask1 = imbinarize(nucMask1 + tempLine);
                          
                        case 3
                            return;
                    end
                case 3
                    return;
            end
            

        end

        compMask1 = imfill(CellMask1, 'holes');
        Lind = find(compMask1 > 0);
        subplot(1,2,2); imshow(CellMask1); hold on;

        stats1 = regionprops(compMask1, 'all');
        nucstats = regionprops(LumenMask1, 'all');

        c = stats1.Centroid;
        text(c(1), c(2), sprintf('%d', dd), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color','red');

        % Finding center of spheroid and creating the lines that will be
        % used to quantify intensity profiles

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

        how_many_point = 6; % 360 / 45;  % number of lines defined in polar coordinates
        coarse_theta = linspace(0, 2*pi, how_many_point + 1);

        xs = round(xc + r*cos(coarse_theta)); % x(0) values
        xs2 = round(xc - r*cos(coarse_theta)); % x(end) values

        ys = round(yc + r*sin(coarse_theta));  % y(0) values
        ys2 = round(yc - r*sin(coarse_theta)); % y(end) values

        Data.CellArea(LumenCnt) = length(Lind);
        Data.LumenArea(LumenCnt) = length(Nind);
        Data.Diameters(LumenCnt,1:2) = [stats1.MajorAxisLength, stats1.MinorAxisLength];
        Data.LumenDia(LumenCnt, 1:2) = [nucstats.MajorAxisLength, nucstats.MinorAxisLength];

        % plotting lines overlaid on the binarized image
        xx = {}; yy = {};
        BasalCorr = [];
        ApicalCorr = [];

        for i = 1:length(xs) % going through each line defined

            m = (yc - ys(i)) / (xc - xs(i));

            if m == Inf     % if the line is vertical
                xx{i}(1:1:r*2) = xs(i);
                yy{i}(1:1:r*2) = floor(linspace(ys(i), yc, r*2));
            elseif m == 0   % if the line is horizontal
                xx{i}(1:1:r*2) = floor(linspace(xs(i), xc, r*2));
                yy{i}(1:1:r*2) = ys(i);
            else            % if the line is angled
                xx{i} = floor(linspace(xs(i), xc, r*2));
                yy{i} = floor( m*(xx{i} - xs(i)) + ys(i) );
            end
            subplot(1,2,2);
            plot(xc, yc, 'r*'); hold on;
            plot(xx{i}, yy{i}); 
        end

        ChInt = {[0],[0]};
        
        % Taking apical versus basolateral intensity measurements

        Data.SumBasalCh(LumenCnt, :) = zeros(1, ImParam.numCh);
        Data.SumApicalCh(LumenCnt, :) = zeros(1, ImParam.numCh);
        Data.AvgBasalCh(LumenCnt, :) = zeros(1, ImParam.numCh);
        Data.AvgApicalCh(LumenCnt, :) = zeros(1, ImParam.numCh);
        Data.MaxBasal(LumenCnt, :) = zeros(1, ImParam.numCh); 
        Data.MaxApical(LumenCnt, :) = zeros(1, ImParam.numCh);

        for i = 1:length(xs)
            if ImParam.nucIdx > 0 % DAPI = only need positions so we use the binarized image
                chIntvals = improfile(nucMask1, [xs(i) xc], [ys(i), yc]);  % basal: apical

                [x_find, ~] = find(chIntvals > 0);
                x_first = x_find(1); x_last = x_find(end);
                x_mid = round(((x_last - x_first)/2) + x_first);

            else
                chIntvals = improfile(CellMask1, [xs(i) xc], [ys(i), yc]);  % basal: apical
                [x_find, ~] = find(chIntvals > 0);
                x_first = x_find(1); x_last = x_find(end);
                x_mid = round(((x_last - x_first)/2) + x_first);
            end
            

            
            for cc = 1:ImParam.numCh % for all other channels, need to use raw image outputs to get intensity profiles
                tempImg = immultiply(Data.ImgFiles{n, cc}, uint8(CellMask1));
                chIntvals = improfile(tempImg, [xs(i) xc], [ys(i), yc]);  % basal: apical

                Data.SumBasalCh(LumenCnt, cc) = Data.SumBasalCh(LumenCnt, cc) + sum(chIntvals(1:x_mid));
                Data.SumApicalCh(LumenCnt, cc) = Data.SumApicalCh(LumenCnt, cc) + sum(chIntvals(x_mid+1:end));
                Data.AvgBasalCh(LumenCnt, cc) = Data.AvgBasalCh(LumenCnt, cc) + mean(chIntvals(1:x_mid));
                Data.AvgApicalCh(LumenCnt, cc) = Data.AvgApicalCh(LumenCnt, cc) + mean(chIntvals(x_mid+1:end));

                % Finding max intensity values
                BasalMax = max(chIntvals(1:x_mid)); 
                if BasalMax > Data.MaxBasal(LumenCnt, cc) 
                    Data.MaxBasal(LumenCnt, cc) = BasalMax; 

                end

                ApicalMax = max(chIntvals(x_mid+1:end));
                if ApicalMax > Data.MaxApical(LumenCnt, cc)
                    Data.MaxApical(LumenCnt, cc) = ApicalMax; 

                end

            end

            % Colocalization analysis if stated:
            if isempty(coloc_idx) == 0
                tempImg1 = immultiply(Data.ImgFiles{n, coloc_idx(1)}, uint8(CellMask1));
                chInt1 = improfile(tempImg1, [xs(i) xc], [ys(i), yc]);  % basal: apical
            
                tempImg2 = immultiply(Data.ImgFiles{n, coloc_idx(2)}, uint8(CellMask1));
                chInt2 = improfile(tempImg2, [xs(i) xc], [ys(i), yc]);  % basal: apical
            
                Btemp = corrcoef(chInt1(1:x_mid), chInt2(1:x_mid));
                Atemp = corrcoef(chInt1(x_mid+1:end), chInt2(x_mid+1:end));
                
                BasalCorr(i) = Btemp(1,2);
                ApicalCorr(i) = Atemp(1,2);
            
            end

        end
        
        Data.colocP(LumenCnt, 1:2) = [mean(BasalCorr); mean(ApicalCorr)];

        % Finding sum, max, and avg intensity of cell, lumen, and ECM
        for cc = 1:ImParam.numCh

            tempImg = immultiply(Data.ImgFiles{n,cc}, uint8(CellMask1));
            Data.sumCell(LumenCnt, cc) = sum(sum(tempImg));
            AvgInd = find(tempImg > 0);
            Data.avgCell(LumenCnt, cc) = Data.sumCell(LumenCnt,cc) / length(AvgInd);
            Data.maxCell(LumenCnt, cc) = max(max(tempImg));

            tempImg2 = immultiply(Data.ImgFiles{n,cc}, uint8(LumenMask1));
            Data.sumLumen(LumenCnt, cc) = sum(sum(tempImg2));
            AvgInd = find(tempImg2 > 0);
            Data.avgLumen(LumenCnt, cc) = Data.sumLumen(LumenCnt,cc) / length(AvgInd);
            Data.maxLumen(LumenCnt, cc) = max(max(tempImg2));

            tempImg3 = immultiply(Data.ImgFiles{n,cc}, uint8(ecmMask));                
            Data.sumECM(LumenCnt, cc) = sum(sum(tempImg3));
            AvgInd = find(tempImg3 > 0);
            Data.avgECM(LumenCnt, cc) = Data.sumECM(LumenCnt,cc) / length(AvgInd);
            Data.maxECM(LumenCnt, cc) = max(max(tempImg3));
        end

        button = menu('Does this look ok?', 'Yes', 'No', 'Exit');
        switch button
            case 1
    
            case 2
                ImParam.delIdx = [ImParam.delIdx, n];
            case 3
                return;
        end

        RemoveStr = strfind(Data.ImgName{n}, 'Image Export');    
        Data.OutputNames{LumenCnt} = [Data.ImgName{n}(1:RemoveStr - 1), 'C', num2str(LumenCnt)];
        LumenCnt = LumenCnt + 1;
    end
    %close(4);
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

%  Goal: Export Data collected from run into an excel file

% Name Sheet Names to separate Data Outputs
sheetNames{1} = 'Run Parameters'; % ImParam Outputs
sheetNames{5} = 'Polar Intensities'; % MaxPolarInt
sheetNames{6} = 'Local Intensities'; % Sum ints at cell, lumen, ecm
sheetNames{7} = 'Spheroid Morphology';

% Sheet 1
writetable(struct2table(ImParam,"AsArray", true), Name, 'Sheet', sheetNames{1},'Range', 'A1');

% Sheet 5
Basal = cell2table(ImParam.chNames); Apical = Basal;
CellOutput = table(Basal, Apical); CellOutput = splitvars(CellOutput);

Sum_Results = [Data.SumBasalCh, Data.SumApicalCh];
writecell({'Sum Intensity'}, Name, 'Sheet', sheetNames{5}, 'Range', 'A1');
writetable(CellOutput, Name, 'Sheet', sheetNames{5}, 'Range', 'B1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{5}, 'Range', 'A3');
writematrix(Sum_Results, Name, 'Sheet', sheetNames{5},  'Range', 'B3');

Avg_Results = [Data.AvgBasalCh, Data.AvgApicalCh];
writecell({'Avg Intensity'}, Name, 'Sheet', sheetNames{5}, 'Range', 'M1');
writetable(CellOutput, Name, 'Sheet', sheetNames{5}, 'Range',  'N1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{5}, 'Range', 'M3');
writematrix(Avg_Results, Name, 'Sheet', sheetNames{5},  'Range', 'N3');

Max_Results = [Data.MaxBasal, Data.MaxApical];
writecell({'Max Intensity'}, Name, 'Sheet', sheetNames{5}, 'Range', 'Y1');
writetable(CellOutput, Name, 'Sheet', sheetNames{5}, 'Range',  'Z1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{5}, 'Range', 'Y3');
writematrix(Max_Results, Name, 'Sheet', sheetNames{5},  'Range', 'Z3');


% Sheet 6
Cell = cell2table(ImParam.chNames); Lumen = Cell; ECM = Cell;
CellOutput = table(Cell, Lumen, ECM); CellOutput = splitvars(CellOutput);

Sum_Results = [Data.sumCell, Data.sumLumen, Data.sumECM];
writecell({'Sum Intensity'}, Name, 'Sheet', sheetNames{6}, 'Range', 'A1');
writetable(CellOutput, Name, 'Sheet', sheetNames{6}, 'Range', 'B1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{6}, 'Range', 'A3');
writematrix(Sum_Results, Name, 'Sheet', sheetNames{6},  'Range', 'B3');


Avg_Results = [Data.avgCell, Data.avgLumen, Data.avgECM];
writecell({'Avg Intensity'}, Name, 'Sheet', sheetNames{6}, 'Range', 'R1');
writetable(CellOutput, Name, 'Sheet', sheetNames{6}, 'Range', 'S1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{6}, 'Range', 'R3');
writematrix(Avg_Results, Name, 'Sheet', sheetNames{6},  'Range', 'S3');


Max_Results = [Data.maxCell, Data.maxLumen, Data.maxECM];
writecell({'Max Intensity'}, Name, 'Sheet', sheetNames{6}, 'Range', 'AI1');
writetable(CellOutput, Name, 'Sheet', sheetNames{6}, 'Range', 'AJ1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{6}, 'Range', 'AI3');
writematrix(Max_Results, Name, 'Sheet', sheetNames{6},  'Range', 'AJ3');


% Sheet 7
sheetNames{7} = 'Spheroid Morphology';
Morph_Results = [Data.CellArea, Data.LumenArea, Data.Diameters, Data.LumenDia, Data.colocP];
writecell({'Cell Area', 'Lumen Area', 'Diameter Major', 'Diameter Minor', 'LumenD Major', 'LumenD Minor', 'ColocPBasal', 'ColocPApical'}, Name, 'Sheet', sheetNames{7}, 'Range', 'B1');
writecell(Data.OutputNames', Name, 'Sheet', sheetNames{7}, 'Range', 'A2');
writematrix(Morph_Results, Name, 'Sheet', sheetNames{7}, 'Range', 'B2');

end