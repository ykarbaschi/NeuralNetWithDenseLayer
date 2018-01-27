function Bias=initBias(NumberOfHiddenLayers,NumberOfHiddenUnits,size,NumberOfClasses)
    Bias={};
    for HLayerInd=2:NumberOfHiddenLayers+1
        Bias(HLayerInd)={ones(NumberOfHiddenUnits,1)*size};
    end
    Bias(NumberOfHiddenLayers+2)={ones(NumberOfClasses,1)*size};