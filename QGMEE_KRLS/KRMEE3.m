function [expansionCoefficient,learningCurve,phi] = ...
    KRMEE3(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor,flagLearningCurve,ker_MCC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KRLS based on MEE
% memeory initialization
[inputDimension,trainSize] = size(trainInput);
testSize = length(testTarget);

expansionCoefficient = zeros(trainSize,1);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

Q_matrix = 1/(forgettingFactor*regularizationFactor*ker_MCC.^2 + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));
expansionCoefficient(1) = Q_matrix*trainTarget(1);
phi = zeros(trainSize-inputDimension+1,1);
eL = zeros(inputDimension,1);
% start training
for n = inputDimension+1:trainSize
    ii = 1:n-1;
    ii2 = n:-1:2;
    h_vector = ker_eval(trainInput(:,n),trainInput(:,ii2),typeKernel,paramKernel);
    z_vector = Q_matrix*h_vector;
    error = trainTarget(n) - h_vector'*expansionCoefficient(ii2);
    for jj = 1 : inputDimension
       %   kk2 = 1: jj -1;
       kk = 1: n - inputDimension + jj - 1;
       kk2 = n: -1 :inputDimension - jj +2;
        %%  该版本为ek 基于a(L)算出来的 另外一个版本为记录每次基于a(k)的ek
        %         eL(jj) = trainTarget(n - inputDimension + jj) - ker_eval(trainInput(:,n - inputDimension + jj),trainInput(:,ii),typeKernel,paramKernel)'*expansionCoefficient(ii);
         %         eL(jj) = trainTarget(n - inputDimension + jj) - ker_eval(trainInput(:,n - inputDimension + jj),trainInput(:,kk2),typeKernel,paramKernel)'*expansionCoefficient(kk2);
        eL(jj) = trainTarget(n - inputDimension + jj) - ker_eval(trainInput(:,n - inputDimension + jj),trainInput(:, kk),typeKernel,paramKernel)'*expansionCoefficient(kk2);
        %%
    end
    e0 = eL(inputDimension);
    phi0 = 0;
    %     phi0 = 5.8;
    for kk = 2:inputDimension
        ek = eL(inputDimension-kk+1);
        %             ek = eL(kk);
        phi0 = phi0 + forgettingFactor^(kk-1) * ker_eval(e0, ek,typeKernel,paramKernel);
    end
    phi(n) = phi0;
    r = 1/(regularizationFactor*forgettingFactor^(n)*ker_MCC.^2 / (phi0)+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
%     Q_tmp = zeros(n,n);
%     Q_tmp(ii,ii) = Q_matrix + z_vector*z_vector'*r;
%     Q_tmp(ii,n) = -z_vector*r;
%     Q_tmp(n,ii) = Q_tmp(ii,n)';
%     Q_tmp(n,n) = r;
%     Q_matrix = Q_tmp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Q_tmp = zeros(n,n);
%     Q_tmp(1,1) = r ;
%     Q_tmp(1,ii+1) = -z_vector'*r;
%     Q_tmp(ii+1,1) = -z_vector *r;
%     Q_tmp(ii+1,ii+1) = Q_matrix + z_vector*z_vector'*r;      
%     Q_matrix = Q_tmp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q_tmp = zeros(n,n);
    Q_tmp(1,1) = r ;
    Q_tmp(1,ii2) = -z_vector'*r;
    Q_tmp(ii2,1) = -z_vector *r;
    Q_tmp(ii2,ii2) = Q_matrix + z_vector*z_vector'*r;      
    Q_matrix = Q_tmp;
    
    
    
    % updating
    expansionCoefficient(1) = r*error;
    expansionCoefficient(ii+1) = expansionCoefficient(ii) - z_vector*expansionCoefficient(n);

%     expansionCoefficient(n) = r*error;
%     expansionCoefficient(ii) = expansionCoefficient(ii) - z_vector*expansionCoefficient(n);
    
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