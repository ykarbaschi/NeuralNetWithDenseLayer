function dsigmoid = dsigmoid(x)
for i=1:size(x,1)
    sigmoid(i,1)=1/(1+exp(-x(i,1)));
    dsigmoid = sigmoid(i,1) * (1- sigmoid(i,1));
end