function s=fun_generate_supsig5(N)
t=0:N-1;
sig(1,:)= ((rem(t,23)-11)/9).^5;                              % 曲线波形 （超高斯）
sig(2,:)= ((7-rem(t,15))/6).^7; 
a=rand(1,N);
sig(2,:)=0.1*randn(1,N).*(sign(a-0.05)/2+1/2)+10*randn(1,N).*(sign(a-0.95)/2+1/2);
% sig(3,:)= ((rem(t,19)-9)/7.5).^5;   
 original_x=rand(1,N);
% sig(3,:)=raylrnd(3,1,N);%%the Rayleigh distribution
sig(3,:)=tan((original_x-1/2)*pi);    %Cauchy
% c=50; mu=1; m=N; n=1;
% sig(3,:)=c/2./(erfcinv(unifrnd(0,1,m,n)).^2)+mu;%Levy

sig(3,:)= ((13-rem(t,27))/11).^7; 
% sig(5,:)= ((7-rem(t,15))/6).^5; 
% [s,M_whiten]=fun_whiten(sig);
Pr=0.3;
% SIR=0.1;sigma = sqrt(1/Pr*10^(-SIR/10));
b = randsrc(1,N,[0 1;1-Pr Pr]);%产生NN*1的0,1向量
    Niose_BG  = randn(1,N);
    BG        = b.*Niose_BG;
Pr=0.1;
b = randsrc(1,N,[0 1;1-Pr Pr]);
    sig(4,:)= b.*raylrnd(3,1,N);
    sig(5,:)=BG;
    [s]=fun_remove_mean_std(sig);
