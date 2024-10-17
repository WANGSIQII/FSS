% the pth of mask
label = imread('E:/experiments/RPT-main_zhu/results/CHAOSPNG/abd_mri_liver_gt.png');
% the pth of raw image
im = imread('E:/experiments/RPT-main_zhu/results/CHAOSPNG/abd_mri_liver_img.png');   

color1 = [1,0,0; 0,1,0; 0,0,1; 1,1,0; 0,1,1];
alpha1 = 0.1;
colorimg1 = drawlabel2image(im,label,color1,alpha1);

color2 = [1,0,0; 0,1,0; 0,0,1; 1,1,0; 0,1,1];
alpha2 = 0.9;
colorimg2 = drawlabel2image(im,label,color2,alpha2);

color3 = [1,1,1; 1,1,1; 1,1,1; 1,1,1; 0,0,1];
alpha3 = 0.8;
colorimg3 = drawlabel2image(im,label,color3,alpha3);

% MATLAB自带的colormap
color4 = jet; % matlab自带
alpha4 = 0.5;
colorimg4 = drawlabel2image(im,label,color4,alpha4);

% the pth of masked image
%imwrite(colorimg3, '.../cmr_masked_gt.png');

% 显示
figure
subplot(2,2,1), imshow(colorimg1)
subplot(2,2,2), imshow(colorimg2)
subplot(2,2,3), imshow(colorimg3)
subplot(2,2,4), imshow(colorimg4)


% cmr 天蓝 深红 黄
% Abd 红 蓝 绿 橘黄
