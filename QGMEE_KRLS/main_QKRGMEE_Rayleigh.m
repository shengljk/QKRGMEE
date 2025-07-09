% 2015j SP Kernel recursive maximum correntropy
% KRLS KRMC KLMS KMC KRMEE
% non-Gaussian Noise  
close all, clear all
clc


forgetting_all = 1;

%  QKRGMEE 参数
alpha_QKRGMEE = 0.15;
beta_QKRGMEE = 1.4;
quan_QKRGMEE = 0.00;
forgetting_QKRGMEE = forgetting_all;




% Num = 300;
Num = 1;

% Data size for training and testing
trainSize = 1000;
testSize = 100;

ensembleLearningCurveKlms = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveKmc = zeros(trainSize,1);
ensembleLearningCurveKMMCC = zeros(trainSize,1);
ensembleLearningCurveKrmc = zeros(trainSize,1);
ensembleLearningCurveKrmee = zeros(trainSize,1);
ensembleLearningCurveKRGMEE = zeros(trainSize,1);
ensembleLearningCurveQKRGMEE = zeros(trainSize,1);
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

[MK30,t]=Mackey_Glass(5000,30);

MK30 = MK30(1000:5000);
inputDimension = 10; 
LL = size(MK30,1);
% varNoise =0.1;
%% 设置噪声
%     v1=randn(1,LL)*0.01; v2=randn(1,LL)*8;
%     rp=rand(1,LL);
%     %    vv = v1 + (rp>0.95).*v2;
%     %% 设置噪声
%      vv = (rp<=0.95).*v1 + (rp>0.95).*v2;   % 脉冲噪声(超高斯)
%   alpha_noise = vv';   
    s_sup=fun_generate_supsig5(LL);
    alpha_noise = s_sup(4,:)*1;                       % Rayleigh noise
% alpha_noise = alpha_stable_noise(1.4,0.1,0,0,length(MK30));

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
paramKernel_MCC = 1;
% paramKernel_Klms = sqrt(2)/2;
paramKernel_Klms = sqrt(2)/2;
paramKernel_Krls = paramKernel_Klms ;
paramKernel_Kmc = paramKernel_Klms ;
paramKernel_KMMCC = paramKernel_Klms ;
paramKernel_Krmc = paramKernel_Klms ;
paramKernel_KRMEE = paramKernel_Klms;
paramKernel_KRGMEE = paramKernel_Klms;
paramKernel_KRQGMEE = paramKernel_Klms;
%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               KLMS 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeKlms1 = 0.15;
flagLearningCurve = 1;

[expansionCoefficientKlms1,learningCurveKlms] = ...
    KLMS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Klms,stepSizeKlms1,flagLearningCurve);
%=========end of KLMS ================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regularizationFactorKrls = 0.3;
forgettingFactorKrls = 1;

% flagLearningCurve = 1;
[expansionCoefficientKrls,learningCurveKrls] = ...
    KRLS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Krls,regularizationFactorKrls,forgettingFactorKrls,flagLearningCurve);

% =========end of KRLS================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRMC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regularizationFactorKrmc = 0.1;
forgettingFactorKrmc = 1;

% flagLearningCurve = 1;
[expansionCoefficientKrmc,learningCurveKrmc] = ...
    KRMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Krmc,regularizationFactorKrmc,forgettingFactorKrmc,flagLearningCurve,paramKernel_MCC);

% % =========end of KRMC================


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              KMMCC

% stepSizeKmc = 2.0;%???
stepSizeKMMCC = 0.2;

[expansionCoefficientKMMCC,learningCurveKMMCC] = ...
    KMMCC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KMMCC,stepSizeKMMCC,flagLearningCurve,paramKernel_MCC);
%% =========end of KMMCC================



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KMC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stepSizeKmc = 2.0;%???
stepSizeKmc = 0.4;

[expansionCoefficientKmc,learningCurveKmc] = ...
    KMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Kmc,stepSizeKmc,flagLearningCurve,paramKernel_MCC);
% % =========end of KMC================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRMEE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 5;
regularizationFactorKrmee = 0.7;
forgettingFactorKrmee = 1;
paramKernel_MEE = 0.4;
% flagLearningCurve = 1;
[expansionCoefficientKrmee,learningCurveKrmee,phi] = ...
    KRMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRMEE,regularizationFactorKrmee,forgettingFactorKrmee,flagLearningCurve,paramKernel_MEE);
% % =========end of KRMEE================
%% KRGMEE
L =5;
regularizationFactorKRGMEE = 10;
forgettingFactorKRGMEE =1;

[expansionCoefficientKRGMEE,learningCurveKRGMEE,phi] = ...
    KRGMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRGMEE,regularizationFactorKRGMEE,forgettingFactorKRGMEE,flagLearningCurve);

%% QKRGMEE
regularizationFactorKrQgmee = L.^2 * 0.01 /2;
[expansionCoefficientKrQgmee,learningCurveKrQgmee,phi_QGMEE] = ...
    QKRGMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRQGMEE,regularizationFactorKrQgmee,forgetting_QKRGMEE,flagLearningCurve,alpha_QKRGMEE,beta_QKRGMEE,quan_QKRGMEE);


ensembleLearningCurveKlms = ensembleLearningCurveKlms + learningCurveKlms/Num;
ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls/Num;
ensembleLearningCurveKrmc = ensembleLearningCurveKrmc + learningCurveKrmc/Num;
ensembleLearningCurveKmc = ensembleLearningCurveKmc + learningCurveKmc/Num;
ensembleLearningCurveKMMCC = ensembleLearningCurveKMMCC + learningCurveKMMCC/Num;
ensembleLearningCurveKrmee = ensembleLearningCurveKrmee + learningCurveKrmee/Num;
ensembleLearningCurveKRGMEE = ensembleLearningCurveKRGMEE + learningCurveKRGMEE/Num;
ensembleLearningCurveQKRGMEE = ensembleLearningCurveQKRGMEE + learningCurveKrQgmee/Num;
end

%%
figure(1),
plot(10*log10(ensembleLearningCurveKlms'),'LineWidth',2)
hold on
plot(10*log10(ensembleLearningCurveKrls'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveKrmc'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveKmc'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveKMMCC'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveKrmee'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveKRGMEE'),'LineWidth',2)
plot(10*log10(ensembleLearningCurveQKRGMEE'),'LineWidth',2)

legend('KLMS','KRLS','KRMC','KMC','KMMCC','KRMEE','KRGMEE','QKRGMEE')

% hold off
xlabel('iteration'),ylabel('MSE')
% set(gca, 'FontSize', 14);
% set(gca, 'FontName', 'Arial');
% grid off
% set(gca, 'YScale','log')

figure (2)
plot (sqrt(varNoise)* alpha_noise(1:1000))

% figure(3)
% plot(phi)
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

disp('>>KMMCC')
mseMean = mean(ensembleLearningCurveKMMCC(end-100:end));
mseStd = std(ensembleLearningCurveKMMCC(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);


disp('>>KRLS')
mseMean = mean(ensembleLearningCurveKrls(end-100:end));
mseStd = std(ensembleLearningCurveKrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRMEE')
mseMean = mean(ensembleLearningCurveKrmee(end-100:end));
mseStd = std(ensembleLearningCurveKrmee(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRGMEE')
mseMean = mean(ensembleLearningCurveKRGMEE(end-100:end));
mseStd = std(ensembleLearningCurveKRGMEE(end-100:end));
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

disp('>>KRGMEE')
mseMean = mean(ensembleLearningCurveKRGMEE(end-100:end));
disp(num2str(10*log10(mseMean)));

disp('====================================')

