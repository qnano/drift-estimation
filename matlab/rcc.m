function [shift_interp, shift_estim, images] = rcc(xy, framenum,timebins, zoom, sigma, maxpairs, usecuda)
fprintf('RCC computation ... \n')
rendersize = int32((max(xy,[],'all'))-1);
area = [rendersize, rendersize];
nframes = max(framenum);
framesperbin = nframes/timebins;

imgshape = area*zoom;
images = zeros(timebins, imgshape(1), imgshape(2));

for k = 1:timebins
    img = single(zeros(imgshape));
    indices =  find(int32(floor(0.5+framenum/framesperbin))==k-1);
    spots = zeros(length(indices), 5);
    spots(:,1) = xy(indices,1) * zoom;
    spots(:, 2) = xy(indices,2) * zoom;
    spots(:, 3) = sigma;
    spots(:, 4) = sigma;
    spots(:, 5) = 1;
    
    if size(spots,1) == 0
        disp('ERROR, NO SPOTS IN BINS!!')
        break
    end
    
    % draw gaussians
    images(k,:,:) = call_dme_lib('gauss',usecuda,  img,size(img,2), size(img,1), spots, size(spots,1));
    
end

[pairsi, pairsj] = find(tril(ones(timebins, timebins),-1));
pairs = [pairsj pairsi];

% include max pairs to be done
images = permute(images, [1 3 2]);
fft_images = fft2(permute(images, [2 3 1]));
fft_images = permute(fft_images, [3 1 2]);
shift = zeros(length(pairs), 2);

%findshift pairs
for i = 1: length(pairs)
   a = pairs(i,1);
   b = pairs(i,2);
   
   fft_conv = permute(conj(fft_images(a,:,:)) .* fft_images(b,:,:), [2 3 1]);
   cc = ifft2(fft_conv);
   cc = abs(fftshift(cc));
   shift(i, :,:) = findshift(cc);
end
% -----

pair_shifts = shift;
A = zeros(length(pairs), timebins);
for i = 1:length(pairs)
   A(i, pairs(i,1)) = 1;
   A(i, pairs(i,2)) = -1;
end
inv = pinv(A);

shift_x = inv * pair_shifts(:,1);
shift_y = inv * pair_shifts(:,2);
shift_y = shift_y - shift_y(1);
shift_x = shift_x - shift_x(1);
shift = -[shift_x shift_y]/zoom;

t = (0.5 + (1:1:timebins)-1)*framesperbin;

shift = shift - mean(shift);
shift_estim = zeros(length(shift),3);
shift_estim(:, [1,2]) = shift;
shift_estim(:,3) = t;
if timebins ~= nframes
        spl_x = interp1(t, shift(:,1), linspace(0, nframes, nframes), 'pchip','extrap');
        spl_y = interp1(t, shift(:,2), linspace(0, nframes, nframes), 'pchip','extrap'); %pchip is nice
        
 
        shift_interp = zeros(nframes,2);
        shift_interp(:,1) = spl_x';
        shift_interp(:,2) = spl_y';
else
        shift_interp = shift;
end

shift_interpx = shift_interp(:,2);
shift_interpy = shift_interp(:,1);
shift_interp = [shift_interpx shift_interpy];

end
