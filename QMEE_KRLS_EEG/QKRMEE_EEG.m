function [expansionCoefficient,learningCurve,phi] = ...
    QKRMEE_EEG(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor,flagLearningCurve,quan_thres,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KRLS based on QGMEE
% memeory initialization
alpha_gmee = 2;           % alpha_gmee = 3，beta_gmee = 1.5  for Rayleigh noise
beta_gmee = 0.68;          % alpha_gmee = 0.2，beta_gmee = 4  for mixed-Gaussian noise
sigma = 1;
quan_thres = 0.00;         % 量化阈值


[~,L] = size(trainInput);
testSize = length(testTarget);
expansionCoefficient = zeros(L,1);

if flagLearningCurve
    learningCurve = zeros(L,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

Q_matrix = 1/((beta_gmee^alpha_gmee/alpha_gmee)*forgettingFactor*regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));     % 论文中Q1
% Q_matrix = 1/((beta_gmee^alpha_gmee/alpha_gmee)*forgettingFactor*regularizationFactor + generalized_ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel,alpha_gmee,beta_gmee));     % 论文中Q1
expansionCoefficient(1) = Q_matrix*trainTarget(1);         % 论文中a1;
phi = zeros(L-K+1,1);
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
    
    [C,M] = quantizer(aprioriErr,quan_thres);
    gg = size(M,2);
    for g = 1:gg
        G = M(g,1)*(1/(sqrt(2*pi)*sigma))*exp(-0.5*(e0-C(gg-g+1,1))^2/(sigma^2));
%         if e0 == C(gg-g+1,1)
%             ei_ej = 0;
%         else
%             ei_ej = (abs(e0-C(gg-g+1,1)))^(alpha_gmee-2);
%         end
        phi0 = phi0 + forgettingFactor^(kk-1) * G;
    end
    
%     for kk = 2:n
%         ek = aprioriErr(n-kk+1);
%         G = (alpha_gmee/2*beta_gmee*(gamma(1/alpha_gmee)))*exp(-1*(abs(e0-ek))^alpha_gmee/lambda_e);
%         ei_ej = (abs(e0-ek))^(alpha_gmee-2);
%         phi0 = phi0 + forgettingFactor^(kk-1) * G*ei_ej;
%     end
    phi(n) = phi0;
    % 要改
    r = 1/(regularizationFactor*forgettingFactor^(n)*(beta_gmee^alpha_gmee/alpha_gmee) / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
    
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
    [C,M] = quantizer(aprioriErr,quan_thres);
    gg = size(M,1);
    e0 =aprioriErr(K);
    for g = 1:gg
        G = M(g,1)*(1/(sqrt(2*pi)*sigma))*exp(-0.5*(e0-C(gg-g+1,1))^2/(sigma^2));
%         if e0 == C(gg-g+1,1)
%             ei_ej = 0;
%         else
%             ei_ej = (abs(e0-C(gg-g+1,1)))^(alpha_gmee-2);
%         end
        phi0 = phi0 + forgettingFactor^(kk-1) * G;
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
    
    r = 1/(regularizationFactor*forgettingFactor^(n)*(beta_gmee^alpha_gmee/alpha_gmee) / phi0+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - h_vector'*z_vector);
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