%Main program for the CNN
clc;
clear all;
close all;
%Load the train and test dataset
%plotInitlearningRate.m - Plotting the learning rate
%An image datastore enables you to store large image data, including data that does not fit in memory, 
%and efficiently read batches of images during training of a convolutional neural network.

depthDatasetPath_train = fullfile('D:\UTD_SEM\Fall_17\DIP\Project\ProjectDataset2','train'); % Change the folder for trainning dataset
depthDatasetPath_test = fullfile('D:\UTD_SEM\Fall_17\DIP\Project\ProjectDataset2','test'); % Change the folder for testing dataset

trainData = imageDatastore(depthDatasetPath_train,'IncludeSubfolders',true,'LabelSource','foldernames');
testData = imageDatastore(depthDatasetPath_test,'IncludeSubfolders',true,'LabelSource','foldernames');

%Display some images from both train and test

% figure;
% perm = randperm(10000,20);
% for i = 1:10
%     subplot(4,5,i);
%     imshow(trainData.Files{perm(i)});
% end
% perm = randperm(140,20);
% for i = 1:10
%     subplot(4,5,i+10);
%     imshow(testData.Files{perm(i)});
% end    
imgSize = size(readimage(trainData,1));
label = countEachLabel(trainData);
labelCount = size(label);




%CNN architecture starts here. Adding one layer by layer
gpuDevice(1);
disp(gpuDevice());
layers = [

    imageInputLayer([imgSize 1])
    %Input layer same as the number of pixels in the input image
    convolution2dLayer(7,32,'Padding',1,'Stride',3)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
	
    convolution2dLayer(5,64,'Padding',1,'Stride',2)%
    batchNormalizationLayer
    reluLayer
 
    convolution2dLayer(3,150,'Padding',1,'Stride',1)%
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    
    fullyConnectedLayer(30)
    dropoutLayer(0.6)
    
    fullyConnectedLayer(25)%
    dropoutLayer(0.6)
    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];	
	
	
%Finding the proper initial learning rate by sweeping
%Using sgdm function and maxEpochs as 15.
%Now Fixed inital learning rate 0.005
i = 500; %200->500
figure;title(strcat('LearningRate of ',num2str(0.00001*i)));
options = trainingOptions('sgdm','MaxEpochs',13, ...
	'InitialLearnRate',0.00001*i,'OutputFcn',@plotInitlearningRate,'ExecutionEnvironment','GPU');  

%% Train the Network Using Training Data
% Train the network you defined in layers, using the training data and the
% training options you defined in the previous steps.
convnet = trainNetwork(trainData,layers,options);
disp(gpuDevice());
save convnet; %Save the trained CNN

disp('Time to test 280 images');
%Use timeit to calculate the correct timings insted of tic and toc.
%tic;
f = @()classify(convnet,testData);
disp(timeit(f));
YTest = classify(convnet,testData);
%YTest = classify(convnet,testData);
%toc;
TTest = testData.Labels;

%% 
%confusion matrix. 

[C,order] = confusionmat(TTest,YTest);

% %Plot the confusion matrix
% 
 figure;imshow(C, [], 'InitialMagnification', 'fit');title(strcat('Confusion Matrix for ',num2str(0.00001*i)));
 colorbar;
 axis on;
% 
% %Set the x axis and y axis label to the classification label we have in the
% %dataset.
% 
 set(gca, 'XTickLabel', char(label.Label));
 set(gca, 'YTickLabel', char(label.Label));
% 
% %Display the confusion matrix in the plot
% 
 textStrings = num2str(C(:),'%d'); % Create strings from the matrix values
 textStrings = strtrim(cellstr(textStrings)); % Remove any space padding
 idx = find(strcmp(textStrings(:), '0'));
 textStrings(idx) = {'   '}; %Replace zeros with blank.. Too many zeros will make the plot not ease to see.
 [x,y] = meshgrid(1:labelCount); %Create x and y coordinates for the strings
 hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');    % Plot the strings in the center
 midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
 textColors = repmat(C(:) <= midValue,1,3);    % Choose white or black for the
%                                                 %   text color of the strings so
%                                                 %   they can be easily seen over
%                                                 %   the background color
 set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text colors

disp(C);

accuracy = sum(YTest == TTest)/numel(TTest);   
X = sprintf('Accuracy: %f',accuracy);
disp(X);

