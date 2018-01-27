function dhyper=dhyper(x)
for i=1:size(x,1)
    dhyper(i,1)=(sech(x(i,1)))^2;
end
end
