clear all
close all
clc
a=alpha_stable_noise(1.5,0.1,0,0,5000);
 plot(a,'b','LineWidth',2);

%xlabel('ʱ�������');
%ylabel('����');
xlabel('iteration');
ylabel('Magnitude');