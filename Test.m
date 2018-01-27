function [Percent]=Test(testInd,Features,Classes,Weights,Bias,nonLinearity,...
    NumberOfHiddenUnits,NumberOfHiddenLayers)
    activation={};
    Z={};
    for i=1:size(testInd,2)
        sample=(Features(testInd(i),:))';
        activation(1)={sample};
        for HLayerInd=2:NumberOfHiddenLayers+2
            Z{HLayerInd}=cell2mat(Weights(HLayerInd))*...
                activation{HLayerInd-1}+...
                cell2mat(Bias(HLayerInd));
            activation{HLayerInd}=sigmoid(Z{HLayerInd});
        end       
    end