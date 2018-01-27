function answer=NonLin(nonLinearity,d,Z)
if (strcmp(nonLinearity,'sigmoid'))
    if(d)
        answer = dsigmoid(Z);
    else
        answer = sigmoid(Z);
    end
elseif (strcmp(nonLinearity,'hyperbolic'))
    if(d)
        answer = dhyper(Z);
    else
        answer = hyper(Z);
    end
elseif (strcmp(nonLinearity,'ReLU'))
    if(d)
        answer = dReLU(Z);
        %answer=softmax(answer);
    else
        answer = ReLU(Z);
        answer=softmax(answer);
    end
end