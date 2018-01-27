function [Weights,WeightsRepos,activationRepos,Bias,trainError,testError,ActTrain,ActTest]=Train(nonLinearity,...
    LearningRate,Momentum,NumberOfIteration,BatchSize,NumberOfHiddenUnits,NumberOfHiddenLayers,Features,Classes,RandomWeights)
% [trainInd,valInd,testInd] = dividerand(size(Features,1),0.8,0,0.2);

n=size(Features,1);
shuffleInd=randperm(n);
trainInd=shuffleInd(1:0.8*n);
testInd=shuffleInd((0.8*n)+1:end);
% temp=[1:n];
% trainInd=temp(1:0.8*n);
% testInd=temp((0.8*n)+1:end);

if BatchSize > size(Features,1)
    warning('BatchSize should be less than or equal to the number of training samples(60% of all samples).');
    return;
end
if (~strcmp(nonLinearity,'sigmoid') && ~strcmp(nonLinearity,'hyperbolic') && ~strcmp(nonLinearity,'ReLU'))
    warning('Use sigmoid or hyperbolic or ReLU as nonLinearity');
    return;
end
if(BatchSize>size(trainInd,2))
    warning('Batch size is greater than Training Set, Then we use all training set as batch size :D');
end

NumberOfClasses = size(unique(Classes),1);
limit=1;
%init Weights
% we should record all weights in all layer  in all iteration we use it for
% drawing graphs
Weights=InitializeWeights(NumberOfHiddenLayers,...
NumberOfHiddenUnits,size(Features,2),limit,NumberOfClasses,RandomWeights);
WeightsRepos={};
activationRepos={};

%Initiliazation of Bias
biasValue=1;
Bias=initBias(NumberOfHiddenLayers,...
NumberOfHiddenUnits,biasValue,NumberOfClasses);

%make tk matrix for sigmoid all zero instaed of that class;
%tk=zeros(NumberOfClasses,1);

basket=floor(size(trainInd,2)/BatchSize);
% for i=1:basket
%     Bucket(i)=trainInd(i+(basket-1)*BatchSize:(basket*BatchSize));
% end
DeltaWeightsPrevious=InitializeWeights(NumberOfHiddenLayers,...
            NumberOfHiddenUnits,size(Features,2),0,NumberOfClasses,RandomWeights);
% x: Set the corresponding activation a1 for the input layer.

for iter=1:NumberOfIteration 
    for b=1:basket
        DeltaWeightsCurrent=InitializeWeights(NumberOfHiddenLayers,...
            NumberOfHiddenUnits,size(Features,2),0,NumberOfClasses,RandomWeights);
        DeltaBias=initBias(NumberOfHiddenLayers,...
            NumberOfHiddenUnits,0,NumberOfClasses);
        
        for batch=(1+((b-1)*BatchSize)):b*BatchSize
            
            [activation,Z]=FeedForward(Features,trainInd,batch,nonLinearity,...
                Classes,NumberOfHiddenLayers,Weights,Bias);
            
            tk=makeTargetVec(nonLinearity,Classes,trainInd,batch);
            
            % calculate other Layers Delta
            [DeltaWeightsCurrent,DeltaBias]=BackPropagation(NumberOfHiddenLayers,tk,Z,Weights,nonLinearity,...
                activation,DeltaWeightsCurrent,DeltaBias,LearningRate);
        end
        %updating Weights
        [Weights,Bias,DeltaWeightsPrevious]=UpdateWeightsAndBias(Weights,Bias,Momentum,...
            DeltaWeightsCurrent,DeltaBias,DeltaWeightsPrevious);        
    end
    [trainError(iter),ActTrain{iter}] = CalcError(trainInd,Weights,Bias,Features,nonLinearity,Classes,NumberOfHiddenLayers);
    [testError(iter),ActTest{iter}]= CalcError(testInd,Weights,Bias,Features,nonLinearity,Classes,NumberOfHiddenLayers);
    WeightsRepos{iter}=Weights;
% check for Overfitting ... If yes Return with matrix Weights and...
end
end

function Weights=InitializeWeights(NumberOfHiddenLayers,NumberOfHiddenUnits,sampleSize,limit,NumberOfClasses,Random)
    Weights={};
    if (Random)
        Weights(2)={initWeights(NumberOfHiddenUnits,sampleSize,limit)};
        Weights(NumberOfHiddenLayers+2)={initWeights(NumberOfClasses,NumberOfHiddenUnits,limit)};
        if(NumberOfHiddenLayers >1)
            for HLayerInd=3:NumberOfHiddenLayers+1
                Weights(HLayerInd)={initWeights(NumberOfHiddenUnits,NumberOfHiddenUnits,limit)};
            end
        end
    else
        Weights(2)={ones(NumberOfHiddenUnits,sampleSize)*limit};
        Weights(NumberOfHiddenLayers+2)={ones(NumberOfClasses,NumberOfHiddenUnits)*limit};
        if(NumberOfHiddenLayers >1)
            for HLayerInd=3:NumberOfHiddenLayers+1
                Weights(HLayerInd)={ones(NumberOfHiddenUnits,NumberOfHiddenUnits)*limit};
            end
        end
    end
end

function [error,Act]=CalcError(dataInd,Weights,Bias,Features,nonLinearity,Classes,NumberOfHiddenLayers)
    error=0;
    Act={};
    for i=1:size(dataInd,2)
        [activation,Z]=FeedForward(Features,dataInd,i,nonLinearity,...
            Classes,NumberOfHiddenLayers,Weights,Bias);
        tk=makeTargetVec(nonLinearity,Classes,dataInd,i);
        diff=(tk-cell2mat(activation(NumberOfHiddenLayers+2)));
        error=error+0.5*(sum(diff.*diff));
        Act{i}=activation;
    end
    error=error/size(dataInd,2);
end

function [Weights,Bias,DeltaWeightsPrevious]=UpdateWeightsAndBias(Weights,Bias,Momentum,...
    DeltaWeightsCurrent,DeltaBias,DeltaWeightsPrevious)
    for i=1:size(Weights,2)
        Weights(i)={cell2mat(Weights(i))+cell2mat(DeltaWeightsCurrent(i))+...
            Momentum*cell2mat(DeltaWeightsPrevious(i))};
        Bias(i)={cell2mat(Bias(i))+cell2mat(DeltaBias(i))};
    end
    DeltaWeightsPrevious=DeltaWeightsCurrent;
end

function [DeltaWeightsCurrent,DeltaBias]=BackPropagation(NumberOfHiddenLayers,tk,Z,Weights,nonLinearity,...
    activation,DeltaWeightsCurrent,DeltaBias,LearningRate)
    
    Delta={};
    Delta(NumberOfHiddenLayers+2)={(tk-cell2mat(activation(NumberOfHiddenLayers+2))).*...
        NonLin(nonLinearity,1,cell2mat(Z(NumberOfHiddenLayers+2)))};
    
    for HLayerInd=(NumberOfHiddenLayers+1):-1:2
        Delta(HLayerInd)={((cell2mat(Weights(HLayerInd+1)))' * cell2mat(Delta(HLayerInd+1))).*...
            NonLin(nonLinearity,1,cell2mat(Z(HLayerInd)))};
    end 
    
    for HLayerInd=(NumberOfHiddenLayers+2):-1:2
        DeltaWeightsCurrent(HLayerInd)={cell2mat(DeltaWeightsCurrent(HLayerInd))+...
            LearningRate*(cell2mat(Delta(HLayerInd)) * (cell2mat(activation(HLayerInd-1)))')};
        DeltaBias(HLayerInd)={cell2mat(DeltaBias(HLayerInd))+LearningRate*cell2mat(Delta(HLayerInd))};
    end
end

function tk=makeTargetVec(nonLinearity,Classes,trainInd,Index)
    NumberOfClasses = size(unique(Classes),1);    
    if(strcmp(nonLinearity,'sigmoid'))
        tk=zeros(NumberOfClasses,1);
        tk(Classes(trainInd(Index)),1)=1;
    elseif(strcmp(nonLinearity,'hyperbolic'))
        tk=ones(NumberOfClasses,1)*-1;
        tk(Classes(trainInd(Index)),1)=1;
    elseif(strcmp(nonLinearity,'ReLU'))
        % how we lmit the ReLU method
        tk=zeros(NumberOfClasses,1);
        tk(Classes(trainInd(Index)),1)=1;
    end
end

%will create a m by n integer matrix with values from 1 to d
function initWeights=initWeights(m,n,d)
for (i=1:m)
    for (j=1:n)
    initWeights(i,j)=rand()*d;
    end
end
end

function [activation,Z]=FeedForward(Features,trainInd,Index,nonLinearity,Classes,NumberOfHiddenLayers,Weights,Bias)
    activation={};    
    sample=(Features(trainInd(Index),:))';
    activation(1)={sample};
    Z={};
    for HLayerInd=2:NumberOfHiddenLayers+2
        Z(HLayerInd)={cell2mat(Weights(HLayerInd))*...
            cell2mat(activation(HLayerInd-1))+...
            cell2mat(Bias(HLayerInd))};
        activation(HLayerInd)={NonLin(nonLinearity,0,cell2mat(Z(HLayerInd)))};
    end
end