function [C,M]=quantizer(e_win,quan_thres)
e_win = e_win';
[~,n] = size(e_win);
Ci = e_win(1,1);
Qei = e_win(1,1);
for i = 2:1:n  
    dis = abs(Ci-e_win(1,i));
    [min_dis,value_dex] = min(dis);
    if min_dis < quan_thres
         Qei(1,i) = Ci(1,value_dex);
    else
        Ci = [Ci,e_win(1,i)];
        Qei(1,i) = e_win(1,i);
    end
end
inf=tabulate(Qei);
C = inf(:,1);
M = inf(:,2);
end


