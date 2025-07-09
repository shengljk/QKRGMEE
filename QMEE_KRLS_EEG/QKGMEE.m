function [expansionCoefficient,learningCurve] = ...
    QKGMEE(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeFeatureVector,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%testInput:     testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:    desired signal for testing testSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
% L: 表示窗口的长度
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:        consisting of coefficients of the kernel expansion
%weightVector:      the linear weight vector
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: none.


% memeory initialization
trainSize = length(trainTarget);
[~,L] = size(trainInput);
testSize = length(testTarget);
alpha_gmee = 2;           % alpha_gmee = 3，beta_gmee = 1.5  for Rayleigh noise
beta_gmee = 1.8;          % alpha_gmee = 0.2，beta_gmee = 4  for mixed-Gaussian noise
quan_thres = 1.0;         % 量化阈值

lambda_e =beta_gmee^alpha_gmee;

expansionCoefficient = zeros(trainSize,1);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

% n=1 init
aprioriErr = zeros(trainSize,1);
% networkOutput  = zeros(trainSize,1);
aprioriErr(1) = trainTarget(1);
expansionCoefficient(1) = stepSizeFeatureVector*aprioriErr(1);
% start
for n = 2:K
    networkOutput = zeros(n,1);
    % training
    % filtering
    ii = 1:n-1;
    for kk = 1:n
        networkOutput(kk) = expansionCoefficient(ii)'* ker_eval(trainInput(:,kk),trainInput(:,ii),typeKernel,paramKernel) ;
    end
    
    phi0 = 0;
    aprioriErr = trainTarget(1:n) - networkOutput; 
    e0 =aprioriErr(n);
    [C,M] = quantizer(aprioriErr,quan_thres);
    gg = size(M,2);
    for iii = 1:n
        ei =aprioriErr(iii);
        for g = 1:gg
            G = M(g,1)*exp(-1*(abs(ei-C(gg-g+1,1)))^alpha_gmee/lambda_e);
            if ei == C(gg-g+1,1)
                ei_ej = 0;
            else
                ei_ej = (abs(ei-C(gg-g+1,1)))^(alpha_gmee-1);
            end
%             phi0 = phi0 + G*ei_ej*sign(ei_ej)*ker_eval(trainInput(:,iii),trainInput(:,iii),typeKernel,paramKernel);
            phi0 = phi0 + G*ei_ej*sign(ei_ej);
        end
    end
    
%     expansionCoefficient(n) = expansionCoefficient(n)+ stepSizeFeatureVector*phi0;
    expansionCoefficient(n) =  stepSizeFeatureVector*phi0;
    
%     networkOutput(n) = expansionCoefficient(ii)'*ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);   % 网络输出
%     aprioriErr(n) = trainTarget(n) - networkOutput(n); 
%     % updating
%     expansionCoefficient(n) = stepSizeFeatureVector*aprioriErr(n);
    
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

networkOutput = zeros(K,1);
for n = K+1:L
    % training
    % filtering
    ii = 1:n-1;
    for kk = 1:K
        networkOutput(kk) = expansionCoefficient(ii)'* ker_eval(trainInput(:,n+kk-K),trainInput(:,ii),typeKernel,paramKernel) ;
    end
    
    phi0 = 0;
    aprioriErr = trainTarget(n-K+1:n) - networkOutput; 
    e0 =aprioriErr(K);
    [C,M] = quantizer(aprioriErr,quan_thres);
    gg = size(M,2);
    for iii = 1:K
        ei =aprioriErr(iii);
        for g = 1:gg
            G = M(g,1)*exp(-1*(abs(ei-C(gg-g+1,1)))^alpha_gmee/lambda_e);
            if ei == C(gg-g+1,1)
                ei_ej = 0;
            else
                ei_ej = (abs(ei-C(gg-g+1,1)))^(alpha_gmee-1);
            end
%             phi0 = phi0 + G*ei_ej*sign(ei_ej)*ker_eval(trainInput(:,iii),trainInput(:,iii),typeKernel,paramKernel);
               phi0 = phi0 + G*ei_ej*sign(ei_ej);
        end
    end
    
%      expansionCoefficient(n) = expansionCoefficient(n)+ stepSizeFeatureVector*phi0;
      expansionCoefficient(n) =  stepSizeFeatureVector*phi0;
    
%     networkOutput(n) = expansionCoefficient(ii)'*ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);   % 网络输出
%     aprioriErr(n) = trainTarget(n) - networkOutput(n); 
%     % updating
%     expansionCoefficient(n) = stepSizeFeatureVector*aprioriErr(n);
    
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end 
return

