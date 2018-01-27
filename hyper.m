function hyper=hyper(x)
for i=1:size(x,1)
    hyper(i,1)=tanh(x(i,1));
end
end
