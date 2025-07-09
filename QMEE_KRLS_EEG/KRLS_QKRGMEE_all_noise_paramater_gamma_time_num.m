% 2015j SP Kernel recursive maximum correntropy
% KRLS KRMC KLMS KMC KRMEE
% EEG data processing   
close all;
clear;
clc;

%% 可调参数
% 公用参数
quan_QKRGMEE0 = 0.01;
quan_QKRGMEE1 = 0.01;
quan_QKRGMEE2 = 0.01;
quan_QKRGMEE3 = 0.01;
quan_QKRGMEE4 = 0.01;
quan_QKRGMEE5 = 0.01;

L0 = 50;                     % for MEE QMEE GMEE QGMEE
L1 = 50;                     % for MEE QMEE GMEE QGMEE
L2 = 50;                     % for MEE QMEE GMEE QGMEE
L3 = 50;                     % for MEE QMEE GMEE QGMEE
L4 = 50;                     % for MEE QMEE GMEE QGMEE
L5 = 50;                     % for MEE QMEE GMEE QGMEE

forgetting_all = 1;         % for RLS-type algorithms
flagLearningCurve = 1;

% RLS 参数
forgetting_KRLS = forgetting_all;

%  QKRGMEE 参数
alpha_QKRGMEE0 = 0.4;    
alpha_QKRGMEE1 = 0.4;    
alpha_QKRGMEE2 = 0.4;    
alpha_QKRGMEE3 = 0.4;    
alpha_QKRGMEE4 = 0.4;   
alpha_QKRGMEE5 = 0.4;    

beta_QKRGMEE0 = 0.1;
beta_QKRGMEE1 = 0.4;
beta_QKRGMEE2 = 2;
beta_QKRGMEE3 = 4;
beta_QKRGMEE4 = 8;
beta_QKRGMEE5 = 30;
% quan_QKRGMEE1 = 0.02;
forgetting_QKRGMEE1 = forgetting_all;
%%

Num = 1;

% Data size for training and testing
trainSize = 1000;
testSize = 100;

ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveKrQgmee0 = zeros(trainSize,1);
ensembleLearningCurveKrQgmee1 = zeros(trainSize,1);
ensembleLearningCurveKrQgmee2 = zeros(trainSize,1);
ensembleLearningCurveKrQgmee3 = zeros(trainSize,1);
ensembleLearningCurveKrQgmee4 = zeros(trainSize,1);
ensembleLearningCurveKrQgmee5 = zeros(trainSize,1);

disp([num2str(Num),' Monte Carlo simulations. Please wait...'])
for kk = 1:Num
    disp(kk);
    
%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load MK30   %MK30 5000*1

% load MK30   %MK30 5000*1

[MK30,t]=Mackey_Glass(5000,30);

MK30 = MK30(1000:5000);
inputDimension = 10; 
LL = size(MK30,1);
% varNoise =0.1;
%% 设置噪声
%%  Mixed-Gaussian noise
%     v1=randn(1,LL)*0.01; v2=randn(1,LL)*8;
%     rp=rand(1,LL);
%     %    vv = v1 + (rp>0.95).*v2;
%     %% 设置噪声
%      vv = (rp<=0.95).*v1 + (rp>0.95).*v2;   % 脉冲噪声(超高斯)
%   alpha_noise = vv';  

%%  Rayleigh noise
%     s_sup=fun_generate_supsig5(LL);
%     alpha_noise = s_sup(4,:)*1;                       % Rayleigh noise
%%  alpha-stable noise 
% alpha_noise = alpha_stable_noise(1.4,0.1,0,0,length(MK30));

%%  mix noise 
    LL = size(MK30,1);
    pro = 0.8;
    pro2 = 0.8;
    v1=randn(1,LL)*0.01; v2=randn(1,LL)*8;
    rp=rand(1,LL);
    %    vv = v1 + (rp>0.95).*v2;
    
    s_sup=fun_generate_supsig5(LL);
    Ray_noise = s_sup(4,:)*1;                       % Rayleigh noise
    vv1 = (rp<=pro).*v1 + (rp>pro).*v2;   % 脉冲噪声(超高斯)
    vv = (rp<=pro2).*vv1 + (rp>pro2).*Ray_noise;
    alpha_noise = vv';

 varNoise = 1;
% L = length (MK30);
% v1=randn(L,1)* 0.1; v2=randn(L,1)*1;
% rp=rand(L,1);
% %    vv = v1 + (rp>0.95).*v2;
% alpha_noise = (rp<=0.95).*v1 + (rp>0.95).*v2;

inputSignal = MK30 + sqrt(varNoise)* alpha_noise;
inputSignal1 = MK30; %noise-free
inputSignal1 = inputSignal1 - mean(inputSignal1);
inputSignal = inputSignal - mean(inputSignal);

%Input training signal with data embedding
trainInput = zeros(inputDimension,trainSize);
for k = 1:trainSize
    trainInput(:,k) = inputSignal(k:k+inputDimension-1);
end

%Input test data with embedding
testInput = zeros(inputDimension,testSize);
for k = 1:testSize
    testInput(:,k) = inputSignal1(k+trainSize:k+inputDimension-1+trainSize);
end

% One step ahead prediction
predictionHorizon = 1;

% Desired training signal
trainTarget = zeros(trainSize,1);
for ii=1:trainSize
    trainTarget(ii) = inputSignal(ii+inputDimension+predictionHorizon-1);
end

% Desired training signal
testTarget = zeros(testSize,1);
for ii=1:testSize
    testTarget(ii) = inputSignal1(ii+inputDimension+trainSize+predictionHorizon-1);
end

%Kernel parameters
typeKernel = 'Gauss';
paramKernel_Klms = sqrt(2)/2;
paramKernel_Krls = paramKernel_Klms;
paramKernel_KRQGMEE = paramKernel_Klms;

%%              KRLS
tic;
regularizationFactorKrls = 0.3;
[expansionCoefficientKrls,learningCurveKrls] = ...
    KRLS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Krls,regularizationFactorKrls,forgetting_KRLS,flagLearningCurve);
toc;
%% QKRGMEE0
tic;
regularizationFactorKrQgmee0 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee0,phi_QGMEE,gg_all0] = ...
    QKRGMEE(L0,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee0,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE0,beta_QKRGMEE0,quan_QKRGMEE0);
toc;

%% QKRGMEE1
tic;
regularizationFactorKrQgmee1 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee1,phi_QGMEE,gg_all1] = ...
    QKRGMEE(L1,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee1,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE1,beta_QKRGMEE1,quan_QKRGMEE1);
toc;
%% QKRGMEE2
tic;
regularizationFactorKrQgmee2 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee2,phi_QGMEE,gg_all2] = ...
    QKRGMEE(L2,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee1,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE2,beta_QKRGMEE2,quan_QKRGMEE2);
toc;
%% QKRGMEE3
tic;
regularizationFactorKrQgmee3 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee3,phi_QGMEE,gg_all3] = ...
    QKRGMEE(L3,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee1,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE3,beta_QKRGMEE3,quan_QKRGMEE3);

%% QKRGMEE4
regularizationFactorKrQgmee4 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee4,phi_QGMEE,gg_all4] = ...
    QKRGMEE(L4,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee1,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE4,beta_QKRGMEE4,quan_QKRGMEE4);
toc;

%% QKRGMEE5
tic;
regularizationFactorKrQgmee5 = L1.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee5,phi_QGMEE,gg_all5] = ...
    QKRGMEE(L5,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee1,forgetting_QKRGMEE1,flagLearningCurve,alpha_QKRGMEE5,beta_QKRGMEE5,quan_QKRGMEE5);
toc;



ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls/Num;
ensembleLearningCurveKrQgmee0 = ensembleLearningCurveKrQgmee0 + learningCurveKrQgmee0/Num;
ensembleLearningCurveKrQgmee1 = ensembleLearningCurveKrQgmee1 + learningCurveKrQgmee1/Num;
ensembleLearningCurveKrQgmee2 = ensembleLearningCurveKrQgmee2 + learningCurveKrQgmee2/Num;
ensembleLearningCurveKrQgmee3 = ensembleLearningCurveKrQgmee3 + learningCurveKrQgmee3/Num;
ensembleLearningCurveKrQgmee4 = ensembleLearningCurveKrQgmee4 + learningCurveKrQgmee4/Num;
ensembleLearningCurveKrQgmee5 = ensembleLearningCurveKrQgmee5 + learningCurveKrQgmee5/Num;

gg_all0_all(kk,:) = gg_all0;
gg_all1_all(kk,:) = gg_all1;
gg_all2_all(kk,:) = gg_all2;
gg_all3_all(kk,:) = gg_all3;
gg_all4_all(kk,:) = gg_all4;
gg_all5_all(kk,:) = gg_all5;


end

%%
figure(1),hold on;
plot(10*log10(ensembleLearningCurveKrls'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee0'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee1'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee2'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee3'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee4'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee5'),'-','LineWidth',2)

legend('KRLS','KRGMEE(L=20, \alpha=0.8, \beta=3.8)','QKRGMEE(L=20, \alpha=0.8, \beta=3.8, \gamma=0.02)','QKRGMEE(L=20, \alpha=0.8, \beta=3.8, \gamma=0.02)','QKRGMEE(L=20, \alpha=0.8, \beta=3.8, \gamma=0.02)','QKRGMEE(L=20, \alpha=0.8, \beta=3.8, \gamma=0.02)');
xlabel('Iteration','FontSize', 14),
ylabel('MSE','FontSize', 14)


%%
disp('>>KRLS')
mseMean = mean(ensembleLearningCurveKrls(end-100:end));
mseStd = std(ensembleLearningCurveKrls(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRLS')
mseMean = mean(ensembleLearningCurveKrls(end-100:end));
disp(num2str(10*log10(mseMean)));


disp('>>KRGMEE0')
mseMean = mean(ensembleLearningCurveKrQgmee0(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all0_all))));
disp('====================================')

disp('>>KRGMEE1')
mseMean = mean(ensembleLearningCurveKrQgmee1(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all1_all))));
disp('====================================')

disp('>>KRGMEE2')
mseMean = mean(ensembleLearningCurveKrQgmee2(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all2_all))));
disp('====================================')

disp('>>KRGMEE3')
mseMean = mean(ensembleLearningCurveKrQgmee3(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all3_all))));
disp('====================================')

disp('>>KRGMEE4')
mseMean = mean(ensembleLearningCurveKrQgmee4(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all4_all))));
disp('====================================')

disp('>>KRGMEE5')
mseMean = mean(ensembleLearningCurveKrQgmee5(end-100:end));
disp(num2str(10*log10(mseMean)));
disp(num2str(mean(mean(gg_all5_all))));
disp('====================================')




