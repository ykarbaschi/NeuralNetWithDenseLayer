function dReLU=dReLU(x)
for i=1:size(x,1)
    dReLU(i,1)=exp(x(i,1))/(exp(x(i,1))+1);
end
end
