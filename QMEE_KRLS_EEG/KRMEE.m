function [expansionCoefficient,learningCurve,phi] = ...
    KRMEE(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor,flagLearningCurve,ker_MCC)
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
phi = zeros(trainSize-K+1,1);
eL = zeros(K,1);
% eL1 = zeros(inputDimension,1);
% start training




for n = 2:K
    networkOutput = zeros(n,1);
    ii = 1:n-1;
    h_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    z_vector = Q_matrix*h_vector;
    for kk = 1:n
        networkOutput(kk) = expansionCoefficient(ii)'* ker_eval(trainInput(:,kk),trainInput(:,ii),typeKernel,paramKernel) ;
    end

    phi0 = 0;
    aprioriErr = trainTarget(1:n) - networkOutput;   
    e0 =aprioriErr(n);
    for kk = 2:n
        ek = aprioriErr(n-kk+1);
        phi0 = phi0 + forgettingFactor^(kk-1) * exp(- (e0-ek).^2/ker_MCC^2);
    end
    phi(n) = phi0;
    r = 1/(regularizationFactor*forgettingFactor^(n)*ker_MCC.^2 / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
    Q_tmp = zeros(n,n);
    Q_tmp(ii,ii) = Q_matrix + z_vector*z_vector'*r;
    Q_tmp(ii,n) = -z_vector*r;
    Q_tmp(n,ii) = Q_tmp(ii,n)';
    Q_tmp(n,n) = r;
    Q_matrix = Q_tmp;
    
    error = trainTarget(n) - h_vector'*expansionCoefficient(ii);
    
    eL(n) =  error;
    
    % updating
    expansionCoefficient(n) = r*error;
    expansionCoefficient(ii) = expansionCoefficient(ii) - z_vector*expansionCoefficient(n);
    
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
for n = K+1:trainSize
    ii = 1:n-1;
    h_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    z_vector = Q_matrix*h_vector;
    error = trainTarget(n) - h_vector'*expansionCoefficient(ii);
    for kk = 1:K
        networkOutput(kk) = expansionCoefficient(ii)'* ker_eval(trainInput(:,n+kk-K),trainInput(:,ii),typeKernel,paramKernel) ;
    end

    phi0 = 0;
    aprioriErr = trainTarget(n-K+1:n) - networkOutput;   
    e0 =aprioriErr(K);
    for kk = 2:K
        ek = aprioriErr(K-kk+1);
        phi0 = phi0 + forgettingFactor^(kk-1) * exp(- (e0-ek).^2/ker_MCC^2);
    end
    phi(n) = phi0;
    
    for kk = 1:K
        eL(kk) = aprioriErr(K-kk+1);
    end
    
    r = 1/(regularizationFactor*forgettingFactor^(n)*ker_MCC.^2 / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
    Q_tmp = zeros(n,n);
    Q_tmp(ii,ii) = Q_matrix + z_vector*z_vector'*r;
    Q_tmp(ii,n) = -z_vector*r;
    Q_tmp(n,ii) = Q_tmp(ii,n)';
    Q_tmp(n,n) = r;
    Q_matrix = Q_tmp;
    % updating
    expansionCoefficient(n) = r*error;
    expansionCoefficient(ii) = expansionCoefficient(ii) - z_vector*expansionCoefficient(n);
    
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