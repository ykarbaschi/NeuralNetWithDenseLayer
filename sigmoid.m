function sigmoid=sigmoid(x)
for i=1:size(x,1)
    sigmoid(i,1)=1/(1+exp(-x(i,1)));
end
end

