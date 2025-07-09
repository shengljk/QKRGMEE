function Y = fun_generate_subsig5( length, graphAndPrint)
% SUBSIG5 ---- generate five supgaussian signals with zero mean and unit variance.
% These signals are taken from following papers:
%   H.H.Yang, Serial updating rule for blind separation derived from the method of 
%   scoring. IEEE Trans. Signal Processing, vol.47, pp.2279-2285, Aug.1999
% Format:  Y = subsig5( length, graphAndPrint)
% Command: 
%          Y = subsig5( 2000, 'on');
%          Y = subsig5( 5000);
% Parameters:
%    < Input >
%        length --- sampling points of sources
% graphAndPrint --- if graphAndPrint == 'on', then graph the generated sources and
%                   print the information of the sources (Default)
%                   if graphAndPrint == 'off', then NOT do so           
%
% Author:    Zhi-Lin Zhang
% E-mail:    zhangzl@vip.163.com
% Version:   1.0
% Update:    Nov.24, 2004


%====================================================================
% Check the parameters and set default value
if(nargin<2)  show = 1;  
else
    switch lower(graphAndPrint)
        case 'on'
            show = 1;
        case 'off'
            show = 0;
        otherwise
            error(sprintf('Illegal value [ %s ] for parameter: ''show''\n', graphAndPrint));
    end
end

%====================================================================
% 产生信号
t=linspace(0,1,length);
Y(1,:) = sign(cos(2*pi*155*t));
Y(2,:) = sin(2*pi*800*t);
Y(3,:) = sin(2*pi*300*t + 6*cos(2*pi*60*t));
% Y(4,:) = sin(2*pi*90*t);

Y(4,:) = sin(rand(1,length)*2*pi);

Y(5,:) = 2*rand(1,length)-1;

Y = fun_remove_mean_std(Y);          % 使均值为0 方差为1

% if(show)    % graph the sources and print the information
%     figure;
%     for i = 5 : -1 : 1
%         subplot(5,1,i);
%         plot(Y(i,[length-499:length]));axis([-inf,inf,-5,5]);
%     end   
%     title('\bf Sources');
%     fprintf('\n');
%     fprintf('================== Generating 5 subguassian sources ... ===============\n');
%     for i =1:5
%         fprintf('      No. %g sources:  kurtosis %g,   skewness %g \n', i, kurtosis(Y(i,:)'), skewness(Y(i,:)'));
%     end
%     fprintf('================== All sources have been generated ! ==================\n');
%     
% end
