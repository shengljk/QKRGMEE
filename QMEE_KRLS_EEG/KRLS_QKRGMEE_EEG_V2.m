% 2015j SP Kernel recursive maximum correntropy
% KRLS KRMC KLMS KMC KRMEE
% EEG data processing   
close all;
clear;
clc;

%% 可调参数
% 公用参数
L = 2;                     % for MEE QMEE GMEE QGMEE
forgetting_all = 1;         % for RLS-type algorithms
flagLearningCurve = 1;

% RLS 参数
forgetting_KRLS = forgetting_all;

% KRMC 参数
sigma_KRMC = 1;
forgetting_KRMC = forgetting_all;

% KRMEE 参数
sigma_KRMEE = 1;
forgetting_KRMEE = forgetting_all;

% QKRMEE 参数
sigma_QKRMEE = 1;
quan_thres_QKRMEE = 0.03;
forgetting_QKRMEE = forgetting_all;

% KRGMEE 参数
alpha_KRGMEE = 1.6;
beta_KRGMEE = 3.8;
forgetting_KRGMEE = forgetting_all;

%  QKRGMEE 参数
alpha_QKRGMEE = 1.6;
beta_QKRGMEE = 3.8;
quan_QKRGMEE = 0.02;
forgetting_QKRGMEE = forgetting_all;
%%

Num = 1;

% Data size for training and testing
trainSize = 1000;
testSize = 100;

ensembleLearningCurveKlms = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveKmc = zeros(trainSize,1);
ensembleLearningCurveKrmc = zeros(trainSize,1);
ensembleLearningCurveKrmee = zeros(trainSize,1);
ensembleLearningCurveQKRMEE = zeros(trainSize,1);
ensembleLearningCurveKrgmee = zeros(trainSize,1);
ensembleLearningCurveKrQgmee = zeros(trainSize,1);

disp([num2str(Num),' Monte Carlo simulations. Please wait...'])
for k = 1:Num
    disp(k);
    
%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load MK30   %MK30 5000*1

load FP1;
FP1 = FP1(10000:end);

% FP1 = FP1(70000:75000)';
% FP1 = FP1(7.35*10^4:7.85*10^4)';
FP1 = FP1(7.202*10^4:7.702*10^4)';
% varNoise =0.1;
inputDimension = 10; 
alpha_noise = alpha_stable_noise(1.4,0.1,0,0,length(FP1));
 
 varNoise = 1;
% L = length (MK30);
% v1=randn(L,1)* 0.1; v2=randn(L,1)*1;
% rp=rand(L,1);
% %    vv = v1 + (rp>0.95).*v2;
% alpha_noise = (rp<=0.95).*v1 + (rp>0.95).*v2;

inputSignal = FP1 + sqrt(varNoise)* alpha_noise;
inputSignal1 = FP1; %noise-free
inputSignal1 = (inputSignal1-min(inputSignal1)*ones(1,size(inputSignal1,2)))/(max(inputSignal1)-min(inputSignal1));
inputSignal = (inputSignal-min(inputSignal)*ones(1,size(inputSignal,2)))/(max(inputSignal)-min(inputSignal));

% inputSignal = FP1 ; 
% inputSignal1 = FP1; %noise-free
% inputSignal1 = (inputSignal1-min(inputSignal1)*ones(1,size(inputSignal1,2)))/(max(inputSignal1)-min(inputSignal1));
% inputSignal = (inputSignal-min(inputSignal)*ones(1,size(inputSignal,2)))/(max(inputSignal)-min(inputSignal));

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
% paramKernel_Klms = sqrt(2)/2;
paramKernel_Klms = sqrt(2)/2;
paramKernel_Krls = paramKernel_Klms;
paramKernel_Kmc = paramKernel_Klms;
paramKernel_Krmc = paramKernel_Klms;
sigma_KRMEE = paramKernel_Klms;
paramKernel_QKRMEE = paramKernel_Klms;
paramKernel_KRGMEE = paramKernel_Klms;
paramKernel_KRQGMEE = paramKernel_Klms;


%%              KRLS
regularizationFactorKrls = 0.3;
[expansionCoefficientKrls,learningCurveKrls] = ...
    KRLS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Krls,regularizationFactorKrls,forgetting_KRLS,flagLearningCurve);

%%              KRMC
regularizationFactorKrmc = 0.1;
[expansionCoefficientKrmc,learningCurveKrmc] = ...
    KRMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Krmc,regularizationFactorKrmc,forgetting_KRMC,flagLearningCurve,sigma_KRMC);
%%              KMC
% [expansionCoefficientKmc,learningCurveKmc] = ...
%     KMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Kmc,stepSizeKMC,flagLearningCurve,sigma_KMC);

%%              KRMEE
regularizationFactorKrmee = L.^2 * 0.01 /2;
[expansionCoefficientKrmee,learningCurveKrmee,phi] = ...
    KRMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,sigma_KRMEE,regularizationFactorKrmee,forgetting_KRMEE,flagLearningCurve,sigma_KRMEE);

%% QKRMEE
regularizationFactorQKRMEE = L.^2 * 0.01 /2;
[expansionCoefficientQKRMEE,learningCurveQKRMEE,phi_QKRMEE] = ...
    QKRMEE_EEG(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_QKRMEE,regularizationFactorQKRMEE,forgetting_QKRMEE,flagLearningCurve,quan_thres_QKRMEE,sigma_QKRMEE);



%% KRLSGMEE
regularizationFactorKrgmee = L.^2 * 0.01 /2;
[expansionCoefficientKrgmee,learningCurveKrgmee,phi] = ...
    KRGMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRGMEE,regularizationFactorKrgmee,forgetting_KRGMEE,flagLearningCurve,alpha_KRGMEE,beta_KRGMEE);
 

%% QKRGMEE
regularizationFactorKrQgmee = L.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee,phi_QGMEE] = ...
    QKRGMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee,forgetting_QKRGMEE,flagLearningCurve,alpha_QKRGMEE,beta_QKRGMEE,quan_QKRGMEE);


ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls/Num;
ensembleLearningCurveKrmc = ensembleLearningCurveKrmc + learningCurveKrmc/Num;
ensembleLearningCurveKrmee = ensembleLearningCurveKrmee + learningCurveKrmee/Num;
ensembleLearningCurveQKRMEE = ensembleLearningCurveQKRMEE + learningCurveQKRMEE/Num;
ensembleLearningCurveKrgmee = ensembleLearningCurveKrgmee + learningCurveKrgmee/Num;
ensembleLearningCurveKrQgmee = ensembleLearningCurveKrQgmee + learningCurveKrQgmee/Num;
end

%%
figure(1),hold on;
plot(10*log10(ensembleLearningCurveKrls'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrmc'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrmee'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveQKRMEE'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrgmee'),'-','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrQgmee'),'-','LineWidth',2)

legend('KRLS','KRMC(\sigma=1.0)','KRMEE(L=10, \sigma=1)','KRQMEE(L=10, \sigma=1, \gamma=0.02)','KRGMEE(L=10, \alpha=1.6, \beta=3.8)','KRQGMEE(L=10, \alpha=1.6, \beta=3.8, \gamma=0.02)')
xlabel('Iteration','FontSize', 14),
ylabel('MSE','FontSize', 14)

%%
disp('====================================')
disp('>>KLMS')
mseMean = mean(ensembleLearningCurveKlms(end-100:end));
mseStd = std(ensembleLearningCurveKlms(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);


disp('>>KMC')
mseMean = mean(ensembleLearningCurveKmc(end-100:end));
mseStd = std(ensembleLearningCurveKmc(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);


disp('>>KRLS')
mseMean = mean(ensembleLearningCurveKrls(end-100:end));
mseStd = std(ensembleLearningCurveKrls(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRMEE')
mseMean = mean(ensembleLearningCurveKrmee(end-100:end));
mseStd = std(ensembleLearningCurveKrmee(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRMC')
mseMean = mean(ensembleLearningCurveKrmc(end-100:end));
mseStd = std(ensembleLearningCurveKrmc(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')

disp('====================================')
disp('>>KLMS')
mseMean = mean(ensembleLearningCurveKlms(end-100:end));
disp(num2str(10*log10(mseMean)));

disp('>>KMC')
mseMean = mean(ensembleLearningCurveKmc(end-100:end));
disp(num2str(10*log10(mseMean)));

disp('>>KRLS')
mseMean = mean(ensembleLearningCurveKrls(end-100:end));
disp(num2str(10*log10(mseMean)));

disp('>>KRMC')
mseMean = mean(ensembleLearningCurveKrmc(end-100:end));
disp(num2str(10*log10(mseMean)));

disp('>>KRMEE')
mseMean = mean(ensembleLearningCurveKrmee(end-100:end));
disp(num2str(10*log10(mseMean)));
disp('====================================')



