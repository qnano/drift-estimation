function [final_value] = findshift(cc)
%findshift
r = 6;
hw = 20;


cc_middle = cc(floor(size(cc,1)/2)-hw : floor(size(cc,1)/2)+hw, floor(size(cc,2)/2)-hw : floor(size(cc,2)/2)+hw); 
[~, max_ind] = max(cc_middle, [], 'all', 'linear');
[peakx, peaky] = ind2sub(size(cc_middle), max_ind);
peakx = peakx +  floor(size(cc,1)/2) - hw;
peaky = peaky +  floor(size(cc,2)/2) - hw;

%clip
if peakx> size(cc,1)-r
    peakx = size(cc,1)-r;
end
if peaky> size(cc,2)-r
    peaky = size(cc,2)-r;
end
if peakx  <r
    peakx = r;
end
if peaky < r
    peaky = r;
end
peak = [peakx, peaky];
roi = cc(peak(1)-r+1: peak(1)+r, peak(2)-r+1:peak(2)+r);


% fit gaussian

W = size(roi,1);
[X, Y] = meshgrid(1:1:size(roi,2),1:1:size(roi,1)); 

img_sum = sum(roi, 'all');
momentX = sum(X.*roi, 'all');
momentY = sum(Y.*roi, 'all');
initial_sigma=2;
bg = min(roi, [], 'all');
I = img_sum - bg * W * W;
xdata = zeros(size(X,1),size(Y,2),2);
xdata(:,:,1) = X;
xdata(:,:,2) = Y;

x0 = [ momentX / img_sum,momentY / img_sum,I,bg,2,0];
x0 =x0(1:5);
xin(6) = 0; 
lb = [0,-W/2,0,-W/2,0];
ub = [realmax('double'),W/2,(W/2)^2,W/2,(W/2)^2];
opts = optimset('Display','off');
[x,~,exitflag,~] = lsqcurvefit(@D2GaussFunction,x0,xdata,roi,lb,ub, opts);

if exitflag == 0 | exitflag == -1 | exitflag == -2 
   fprintf('!ERROR! lsqcurvefit in findshift Exitflag is %d, fit not correct',exitflag);  
    
end
x0 = x(2);
y0 = x(4);

final_value = [peak(2)+x0-r-1-size(cc,2)/2, peak(1)+y0-1-r-size(cc,1)/2];

end

