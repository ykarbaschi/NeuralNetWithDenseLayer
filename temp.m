iter=100;

%basket=floor(size(trainInd,2)/BatchSize);
basket=40;
for k=1:iter 
    x=WeightsRepos(k,1:end,2);
    for i=1:size(x,2)
        y{i}=x{1,i}(1,:);
    end
    for i=1:size(y,2)-1
        c(i+((k-1)*basket))=y{i}*y{i+1}';
    end
end
%plot(c);