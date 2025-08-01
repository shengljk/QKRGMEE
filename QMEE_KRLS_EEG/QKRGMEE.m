function [expansionCoefficient,learningCurve,phi,gg_all] = ...
    QKRGMEE(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor,flagLearningCurve,alpha_qgmee,beta_qgmee,quan_qgmee)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KRLS based on QGMEE
% memeory initialization
% alpha_qgmee = 1.5;           % alpha_gmee = 3，beta_gmee = 1.5  for Rayleigh noise
% beta_qgmee = 3.8;          % alpha_gmee = 0.2，beta_gmee = 4  for mixed-Gaussian noise
% quan_qgmee = 0.03;         % 量化阈值

lambda_e =beta_qgmee^alpha_qgmee;
[~,L] = size(trainInput);
testSize = length(testTarget);

expansionCoefficient = zeros(L,1);

if flagLearningCurve
    learningCurve = zeros(L,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

Q_matrix = 1/((beta_qgmee^alpha_qgmee/alpha_qgmee)*forgettingFactor*regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));     % 论文中Q1
% Q_matrix = 1/((beta_gmee^alpha_gmee/alpha_gmee)*forgettingFactor*regularizationFactor + generalized_ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel,alpha_gmee,beta_gmee));     % 论文中Q1
expansionCoefficient(1) = Q_matrix*trainTarget(1);         % 论文中a1;
phi = zeros(L-K+1,1);
eL = zeros(K,1);
% eL1 = zeros(inputDimension,1);
% start training
gg_all = zeros(1,K);
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
    
    [C,M] = quantizer(aprioriErr,quan_qgmee);
    gg = size(M,2);
    gg_all(1,n) = gg;
    for g = 1:gg
        G = M(g,1)*(alpha_qgmee/2*beta_qgmee*(gamma(1/alpha_qgmee)))*exp(-1*(abs(e0-C(gg-g+1,1)))^alpha_qgmee/lambda_e);
        if e0 == C(gg-g+1,1)
            ei_ej = 0;
        else
            ei_ej = (abs(e0-C(gg-g+1,1)))^(alpha_qgmee-2);
        end
        phi0 = phi0 + forgettingFactor^(kk-1) * G*ei_ej;
    end
    
%     for kk = 2:n
%         ek = aprioriErr(n-kk+1);
%         G = (alpha_gmee/2*beta_gmee*(gamma(1/alpha_gmee)))*exp(-1*(abs(e0-ek))^alpha_gmee/lambda_e);
%         ei_ej = (abs(e0-ek))^(alpha_gmee-2);
%         phi0 = phi0 + forgettingFactor^(kk-1) * G*ei_ej;
%     end
    phi(n) = phi0;
    % 要改
    r = 1/(regularizationFactor*forgettingFactor^(n)*(beta_qgmee^alpha_qgmee/alpha_qgmee) / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
    
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
for n = K+1:L
    ii = 1:n-1;
    h_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    z_vector = Q_matrix*h_vector;
    error = trainTarget(n) - h_vector'*expansionCoefficient(ii);

    for kk = 1:K
        networkOutput(kk) = expansionCoefficient(ii)'* ker_eval(trainInput(:,n+kk-K),trainInput(:,ii),typeKernel,paramKernel) ;
    end

    phi0 = 0;
    aprioriErr = trainTarget(n-K+1:n) - networkOutput;   
%     e0 =aprioriErr(K);
    
    % 量化
    [C,M] = quantizer(aprioriErr,quan_qgmee);
    gg = size(M,1);
    gg_all(1,n) = gg;
    e0 =aprioriErr(K);
    for g = 1:gg
        G = M(g,1)*(alpha_qgmee/2*beta_qgmee*(gamma(1/alpha_qgmee)))*exp(-1*(abs(e0-C(gg-g+1,1)))^alpha_qgmee/lambda_e);
        if e0 == C(gg-g+1,1)
            ei_ej = 0;
        else
            ei_ej = (abs(e0-C(gg-g+1,1)))^(alpha_qgmee-2);
        end
        phi0 = phi0 + forgettingFactor^(kk-1) * G*ei_ej;
    end
    
%     phi0 = 0;
%     for kk = 2:K
%         ek = aprioriErr(K-kk+1);
%         G = (alpha_gmee/2*beta_gmee*(gamma(1/alpha_gmee)))*exp(-1*(abs(e0-ek))^alpha_gmee/lambda_e);
%         ei_ej = (abs(e0-ek))^(alpha_gmee-2);
%         phi0 = phi0 + forgettingFactor^(kk-1) * G*ei_ej;
%     end
    phi(n) = phi0;
    
    for kk = 1:K
        eL(kk) = aprioriErr(K-kk+1);
    end
    
    r = 1/(regularizationFactor*forgettingFactor^(n)*(beta_qgmee^alpha_qgmee/alpha_qgmee) / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
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