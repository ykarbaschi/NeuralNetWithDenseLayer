function sgnFun=sgnFun(weights, features)
    if (size(weights,2)~=size(features,1))
        warn('Weights and Features ca not be multiplied');
        return;
    end
    if (features(1)~=1)% check for X0 feature and put it 1
        vertcat(1,features);
    end
    sgnFun = sign(weights * features);