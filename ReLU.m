function ReLU=ReLU(x)
for i=1:size(x,1)
    ReLU(i,1)=log(1+exp(x(i,1)))/log(exp(1));
end
end
