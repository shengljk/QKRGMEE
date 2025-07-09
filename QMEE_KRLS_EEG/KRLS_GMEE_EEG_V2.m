% 2015j SP Kernel recursive maximum correntropy
% KRLS KRMC KLMS KMC KRMEE
% EEG data processing   
close all, clear all
clc

% Num = 300;
Num = 1;

% Data size for training and testing
trainSize = 500;
testSize = 100;

ensembleLearningCurveKlms = zeros(trainSize,1);
ensembleLearningCurveQKGMEE = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveKmc = zeros(trainSize,1);
ensembleLearningCurveKrmc = zeros(trainSize,1);
ensembleLearningCurveKrmee = zeros(trainSize,1);
ensembleLearningCurveKrgmee = zeros(trainSize,1);

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

% inputSignal = FP1 + sqrt(varNoise)* alpha_noise;
% inputSignal1 = FP1; %noise-free
% inputSignal1 = (inputSignal1-min(inputSignal1)*ones(1,size(inputSignal1,2)))/(max(inputSignal1)-min(inputSignal1));
% inputSignal = (inputSignal-min(inputSignal)*ones(1,size(inputSignal,2)))/(max(inputSignal)-min(inputSignal));

inputSignal = FP1 ; 
inputSignal1 = FP1; %noise-free
inputSignal1 = (inputSignal1-min(inputSignal1)*ones(1,size(inputSignal1,2)))/(max(inputSignal1)-min(inputSignal1));
inputSignal = (inputSignal-min(inputSignal)*ones(1,size(inputSignal,2)))/(max(inputSignal)-min(inputSignal));

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
paramKernel_QKGMEE = paramKernel_Klms;
paramKernel_Krls = paramKernel_Klms ;
paramKernel_Kmc = paramKernel_Klms ;
paramKernel_Krmc = paramKernel_Klms ;
paramKernel_KRMEE = paramKernel_Klms ;
paramKernel_KRGMEE = paramKernel_Klms ;
%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               KLMS 
stepSizeKlms1 = 0.15;
flagLearningCurve = 1;

[expansionCoefficientKlms1,learningCurveKlms] = ...
    KLMS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Klms,stepSizeKlms1,flagLearningCurve);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               QKGMEE 
stepSizeQKGMEE = 0.15;
flagLearningCurve = 1;

[expansionCoefficientGKGMEE,learningCurveQKGMEE] = ...
    QKGMEE(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_QKGMEE,stepSizeQKGMEE,flagLearningCurve);


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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KMC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stepSizeKmc = 2.0;%???
stepSizeKmc = 0.2;

[expansionCoefficientKmc,learningCurveKmc] = ...
    KMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_Kmc,stepSizeKmc,flagLearningCurve,paramKernel_MCC);
% % =========end of KMC================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRMEE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 10;
regularizationFactorKrmee = L.^2 * 0.01 /2;
% regularizationFactorKrmee = 0.2;
forgettingFactorKrmee = 1;

paramKernel_MCC = 1;
% flagLearningCurve = 1;
[expansionCoefficientKrmee,learningCurveKrmee,phi] = ...
    KRMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRMEE,regularizationFactorKrmee,forgettingFactorKrmee,flagLearningCurve,paramKernel_MCC);
% % =========end of KRMEE================

% % =========begin of KRLSGMEE================
L = 10;
regularizationFactorKrgmee = L.^2 * 0.01 /2;
% regularizationFactorKrmee = 0.2;
forgettingFactorKrgmee = 1;

paramKernel_MCC = 1;
% flagLearningCurve = 1;
[expansionCoefficientKrgmee,learningCurveKrgmee,phi] = ...
    KRGMEE(L,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel_KRGMEE,regularizationFactorKrgmee,forgettingFactorKrgmee,flagLearningCurve);
% % =========end of KRLSGMEE================


ensembleLearningCurveKlms = ensembleLearningCurveKlms + learningCurveKlms/Num;
ensembleLearningCurveQKGMEE = ensembleLearningCurveQKGMEE + learningCurveQKGMEE/Num;
ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls/Num;
ensembleLearningCurveKrmc = ensembleLearningCurveKrmc + learningCurveKrmc/Num;
ensembleLearningCurveKmc = ensembleLearningCurveKmc + learningCurveKmc/Num;
ensembleLearningCurveKrmee = ensembleLearningCurveKrmee + learningCurveKrmee/Num;
ensembleLearningCurveKrgmee = ensembleLearningCurveKrgmee + learningCurveKrgmee/Num;
end

%%
figure(1),
plot(10*log10(ensembleLearningCurveKlms'),'m-','LineWidth',2)
plot(10*log10(ensembleLearningCurveQKGMEE'),'-','LineWidth',2)
hold on
plot(10*log10(ensembleLearningCurveKrls'),'g:','LineWidth',2)
plot(10*log10(ensembleLearningCurveKrmc'),'r','LineWidth',1)
% plot(10*log10(ensembleLearningCurveKmc'),'b','LineWidth',1)
plot(10*log10(ensembleLearningCurveKrmee'),'k','LineWidth',1)
plot(10*log10(ensembleLearningCurveKrgmee'),'c','LineWidth',1)

legend('KLMS','QKGMEE','KRLS','KRMC','KRMEE','KRGMEE')

% hold off
xlabel('iteration'),ylabel('MSE')
% set(gca, 'FontSize', 14);
% set(gca, 'FontName', 'Arial');
% grid off
% set(gca, 'YScale','log')
% 
% figure (2)
% plot (sqrt(varNoise)* alpha_noise(1:1000))

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

