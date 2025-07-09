function x=fun_remove_mean_std(x);
[row,col]=size(x);
for i=1:row
    tmp_mean=mean(x(i,:));
    x(i,:)=x(i,:)-tmp_mean;
    x(i,:)=x(i,:)/std(x(i,:));
end