clear;clc;

[att1,att2,att3,att4,Classes]=importfile('iris.data',1,150);
att1= Normalize(att1);
att2= Normalize(att2);
att3= Normalize(att3);
att4= Normalize(att4);
Features=horzcat(att1,att2,att3,att4);
Classes = vertcat(ones(50,1),ones(50,1)*2,ones(50,1)*3);

% sizeOfClass=699;
% [att0,att1,att2,att3,att4,att5,att6,att7,att8,att9,Classes]=importCancer('BreastCancer.data',1,sizeOfClass);
%  att1= Normalize(att1);
%  att2= Normalize(att2);
%  att3= Normalize(att3);
%  att4= Normalize(att4);
%  att5= Normalize(att5);
%  att6= Normalize(att6);
%  att7= Normalize(att7);
%  att8= Normalize(att8);
%  att9= Normalize(att9);
%  Features=horzcat(att1,att2,att3,att4,att5,att6,att7,att8,att9);
%  
%  Classes = Classes/2;

NumberOfHiddenUnits=3;
NumberOfHiddenLayers=1;
NumberOfIteration =100;
nonLinearity = 'sigmoid';
BatchSize=3;
LearningRate=0.01;
Momentum=0.3;
RandomWeights=1;

% for histogram in each iteration activation
LayerNum=2;
UnitNum=2;
iterNum=99;

% Divide all samples into Train , Test and Validate with 0.6, 0.2, 0.2
% ratio

[Weights,WeightsRepos,activationRepos,Bias,trainError,testError,ActTrain,ActTest]=Train(nonLinearity,LearningRate,Momentum,NumberOfIteration,BatchSize,...
NumberOfHiddenUnits,NumberOfHiddenLayers,Features,Classes,RandomWeights);

PlotHist(LayerNum,UnitNum,iterNum,ActTrain);
title('Activation Value Histogram');
xlabel('Activation Value');
ylabel('Number of Occurance');
figure
plot(trainError,'-r');
title('Training Versus Test Error');
xlabel('Iteration');
ylabel('error');
hold on
plot(testError,'-b');
legend('train', 'test');

angle= PlotWeightChanges(LayerNum,UnitNum,WeightsRepos);

figure;
plot(angle);
title('Weight Changes');
xlabel('Iteration');
ylabel('angle between wait vectors');

%Test(testInd,Features,Classes,Weights,Bias,nonLinearity,NumberOfHiddenUnits,NumberOfHiddenLayers);

function angle=PlotWeightChanges(LayerNum,UnitNum,WeightRepos)
z={};
angle=[];
for i=1:size(WeightRepos,2)
        x=WeightRepos{1,i};
        y=x{1,LayerNum};
        z{i}=y(UnitNum,:);
end
    for i=1:size(z,2)-1
        angle(i)=GetAngle(z{i},z{i+1});
    end
end

function PlotHist(NumberOfLayer,NumberOfUnit,iter,activation)
    ActInIter=activation{1,iter};
    z=[];
    for i=1:size(ActInIter,2)
        x=ActInIter{1,i};
        y=x{1,NumberOfLayer};
        z(i)=y(NumberOfUnit);
    end
    figure
    hist(z);
end