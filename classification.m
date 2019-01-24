clc;
clear;
for loopWithTimeStep = 20:20


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FROM index 1-50000 %%%%%
% Create T1 and T2. T1:cell format(9*max); T2:categorical format %%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('data4noNaN.mat');
T1 = {};
T2 = {};
previousdate = T6(1,1);
cellIndex = 1;
internIndex = 1;
for i = 1:50000
    
   if(T6(1,i)==previousdate)
       for k = 1:internIndex
            for j = 1:9
               %  T1{cellIndex,1}(j-3,k) = T5(j,i-internIndex+k);
                 
                    T1{cellIndex,1}(j,k) = T6(j+3,i-internIndex+k);
                 
            end
            flaretest =  num2str(T6(21,i));
       end
elseif(T6(1,i)~=previousdate)
    internIndex = 1;
    for j = 1:9
                 
                    T1{cellIndex,1}(j,1) = T6(j+3,i-internIndex+1);
                 
    end
    flaretest =  num2str(T6(21,i));
   end
   T2 = [T2;flaretest];
           
   previousdate = T6(1,i);
    internIndex = internIndex +1;
    cellIndex = cellIndex+1;
end

T2 = categorical(T2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FROM index 1-50000 %%%%%
% Create new T1. T1: cell format.( 9 * want )format

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1want = {};
want = loopWithTimeStep; %%%%%%%%%%%%%%%% 9 * want format
for i = 1:length(T1)
  
   sizei = size(T1{i});
   sizei = sizei(2);
   
   if(sizei>=want)
       
       for j = 1:9
       %T1two{i} = celli([1:9],
       for k = 1:want
        T1want{i}(j,k) = T1{i}(j,sizei-want+k); 
       end
       end
       
   elseif(sizei<want)
       for a = 1:sizei
           for b = 1:9
             T1want{i}(b,a) = T1{i}(b,a);
           end
       end
   end
end
T1want = T1want';






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create random XTrain from index 1-50000, and gain YTrain.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
load('data4noNaN.mat');
T1_2 = {};
T2_2 = {};
markNum = 0;
index = 1;

T1 = T1want;

for i = 1:50000
   
   mark = T6(21,i);
   
   if(mark == 1)
       markNum = markNum+1;
       T1_2{index} = T1{i};
       index = index+1;
       markValue = num2str(1);
       T2_2 = [T2_2;markValue];
   end
   
    if(mark == 0)
        if(markNum>0)
         randomindex = round(rand(1,1)*49999+1);
            if(T6(21,randomindex) == 0)
         T1_2{index} = T1{randomindex};
          markValue = num2str(0);
         T2_2 = [T2_2;markValue];
         markNum = markNum-1;

         index = index+1;
            end
        end
    end
    
end 
classificationXTrain = T1_2';               %Create XTrain data
classificationYTrain = categorical(T2_2);   %Create YTrain data



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FROM index 50001-62580 %%%%%
% Create T1 and T2. T1:cell format(9*max); T2:categorical format %%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
load('data4noNaN.mat');
T1 = {};
T2 = {};
previousdate = T6(1,1);
cellIndex = 1;
internIndex = 1;
for i = 50001:62580
    
   if(T6(1,i)==previousdate)
       for k = 1:internIndex
            for j = 1:9
               %  T1{cellIndex,1}(j-3,k) = T5(j,i-internIndex+k);
                 
                    T1{cellIndex,1}(j,k) = T6(j+3,i-internIndex+k);
                 
            end
            flaretest =  num2str(T6(21,i));
       end
elseif(T6(1,i)~=previousdate)
    internIndex = 1;
    for j = 1:9
                 
                    T1{cellIndex,1}(j,1) = T6(j+3,i-internIndex+1);
                 
    end
    flaretest =  num2str(T6(21,i));
   end
   T2 = [T2;flaretest];
           
   previousdate = T6(1,i);
    internIndex = internIndex +1;
    cellIndex = cellIndex+1;
end

T2 = categorical(T2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FROM index 50001-62580 %%%%%
% Create new T1. T1: cell format.( 9 * want )format

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1want = {};
want = loopWithTimeStep; %%%%%%%%%%%%%%%% 9 * want format
for i = 1:length(T1)
  
   sizei = size(T1{i});
   sizei = sizei(2);
   
   if(sizei>=want)
       
       for j = 1:9
       %T1two{i} = celli([1:9],
       for k = 1:want
        T1want{i}(j,k) = T1{i}(j,sizei-want+k); 
       end
       end
       
   elseif(sizei<want)
       for a = 1:sizei
           for b = 1:9
             T1want{i}(b,a) = T1{i}(b,a);
           end
       end
   end
end
T1want = T1want';
classificationXTest = T1want;  % Create XTest data
classificationYTest = T2;      % Create YTest data




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%      Training      %%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;

layers = [ ...
    sequenceInputLayer(9)%172')
  % reluLayer
    lstmLayer(20,'OutputMode','last','InputWeightsLearnRateFactor',1)   %'OutputMode','sequence'
    %fullyConnectedLayer(5)
    %reluLayer
    fullyConnectedLayer(2)%2
    %reluLayer
    softmaxLayer
    classificationLayer];


%net.Layers(2,1).InputWeightsLearnRateFactor=2;
%'CellState',randn(4,1),...

options = trainingOptions('sgdm', ...
    'Momentum' ,0.95,...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',128,...
    'ValidationFrequency',45,...
    'LearnRateDropFactor',0.5, ...
     'SequenceLength','longest', ...
    'Verbose',1, ...
    'Shuffle','once',...
    'ExecutionEnvironment','gpu',...
     'GradientThreshold',3, ... 
    'Plots','training-progress');

net = trainNetwork(classificationXTrain,classificationYTrain,layers,options);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%      Testing      %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
YPred = classify(net,classificationXTest, ...
    'MiniBatchSize',128, ...
    'SequenceLength','longest');
%acc = sum(YPred == T2_6)./numel(T2_6);
%%%
%[YTest_eit,err_eit]  = classify(net,xtrain_temp);
testLabel_eit=double(YPred)-1;
tureLabel_eit=double(classificationYTest)-1;

TP_eit=length(find( tureLabel_eit==1 & testLabel_eit==1  ));
TN_eit=length(find( tureLabel_eit==0 & testLabel_eit==0  ));
FP_eit=length(find( tureLabel_eit==0 & testLabel_eit==1  ));
FN_eit=length(find( tureLabel_eit==1 & testLabel_eit==0  ));

TPrate_eit=TP_eit/(TP_eit+FN_eit)
TNrate_eit=TN_eit/(TN_eit+FP_eit)
end