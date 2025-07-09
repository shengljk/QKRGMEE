function [expansionCoefficient,learningCurve] = ...
    KMC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeFeatureVector,flagLearningCurve,paramKernel_MCC)
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
testSize = length(testTarget);

expansionCoefficient = zeros(trainSize,1);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

% n=1 init
aprioriErr = zeros(trainSize,1);
networkOutput  = zeros(trainSize,1);
aprioriErr(1) = trainTarget(1);
expansionCoefficient(1) = stepSizeFeatureVector * aprioriErr(1) * exp(-aprioriErr(1).^2/2/paramKernel.^2);
% start
for n = 2:trainSize
    % training
    % filtering
    ii = 1:n-1;
    networkOutput(n) = expansionCoefficient(ii)'*ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    aprioriErr(n) = trainTarget(n) - networkOutput(n);
    % updating
    expansionCoefficient(n) = stepSizeFeatureVector*aprioriErr(n)* exp(-aprioriErr(n).^2/2/paramKernel_MCC.^2);
     
    
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

