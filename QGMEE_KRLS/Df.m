function y=Df(x)
%此函数用于构造麦克-格拉斯(Mackey-Glass)混沌延迟微分方程的形式
a=0.2; 
y=a*x/(1+x^10);
end