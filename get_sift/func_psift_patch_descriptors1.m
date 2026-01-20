function [d] = func_psift_patch_descriptors1(patch)
    % date: 2025.05.12 name: Li Dong
    % patch大小 64*64

    % 设置VLFeat库的路径并初始化
    VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
    addpath(fullfile(VLFEAT_PATH, 'toolbox'));
    vl_setup;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    expphotop='/media/dataset_maker_matlab_python/get_sift/aimages/4.jpg';
    img = imread(expphotop);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    [rows, cols] = size(img);
    centerRow = round(rows / 2);
    centerCol = round(cols / 2);
    startRow = centerRow - 32;
    startCol = centerCol - 32;
    patch = img(startRow:startRow+63, startCol:startCol+63);
    imwrite(patch, '/media/dataset_maker_matlab_python/get_sift/aimages/patch.png');
    % 显示原始图像和标记patch位置
    figure;
    imshow(img);
    title('Original Image with Patch Location');

    % 标记patch位置
    hold on;
    rectangle('Position', [startCol, startRow, 64, 64], 'EdgeColor', 'r', 'LineWidth', 2);
    hold off;

    % 保存标记后的图像
    frame = getframe(gca);
    imwrite(frame.cdata, '/media/dataset_maker_matlab_python/get_sift/aimages/marked_image.png');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % 定义关键点的位置、尺度和方向
    f = [31;31; 1.5; 0]; % 4x1 向量

    % 对图像进行高斯平滑
    I_       = vl_imsmooth(im2double(patch), sqrt(f(3)^2 - 0.5^2)) ;
    [Ix, Iy] = vl_grad(I_) ;
    % 保存Ix和Iy

    mod      = sqrt(Ix.^2 + Iy.^2) ;
    ang      = atan2(Iy,Ix) ;
    grd      = shiftdim(cat(3,mod,ang),2) ;
    grd      = single(grd) ;
    d        = vl_siftdescriptor(grd, f,'magnif', 6,'numSpatialBins', 8) ;

   % I_1       = vl_imsmooth(im2double(patch), 2) ;
  %  [Ix1, Iy1] = vl_grad(I_1) ;
    % 保存Ix和Iy

 %   mod1      = sqrt(Ix1.^2 + Iy1.^2) ;
 %   ang1      = atan2(Iy1,Ix1) ;
 %   grd1      = shiftdim(cat(3,mod1,ang1),2) ;
 %   grd1      = single(grd1) ;
 %   d1        = vl_siftdescriptor(grd1, f) ;


    disp(size(d))
    % 保存Ix和Iy
    figure;
    imagesc(Ix);
    colormap gray;
    axis image;
    title('Gradient Ix');
    imwrite(getframe(gca).cdata, '/media/dataset_maker_matlab_python/get_sift/aimages/Ix.png');

    % 保存Ix和Iy
  %  figure;
  %  imagesc(Ix1);
  %  colormap gray;
 %   axis image;
 %   title('Gradient Ix1');
 %   imwrite(getframe(gca).cdata, '/media/dataset_maker_matlab_python/get_sift/aimages/Ix1.png');


    figure;
    imagesc(Iy);
    colormap gray;
    axis image;
    title('Gradient Iy');
    imwrite(getframe(gca).cdata, '/media/dataset_maker_matlab_python/get_sift/aimages/Iy.png');


    figure;
    imagesc(Iy1);
    colormap gray;
    axis image;
    title('Gradient Iy1');
    imwrite(getframe(gca).cdata, '/media/dataset_maker_matlab_python/get_sift/aimages/Iy1.png');


    %
    figure;
    image(patch); % 显示patch图像
    hold on;
    h1 = vl_plotframe(f);
    h2 = vl_plotframe(f);
    set(h1, 'color', 'k', 'linewidth', 3);
    set(h2, 'color', 'y', 'linewidth', 2);
    h3 = vl_plotsiftdescriptor(d, f);
    set(h3, 'color', 'g'); % 设置描述子的颜色为绿色
    hold off;
    % 保存图像
    saveas(gcf, '/media/dataset_maker_matlab_python/get_sift/aimages/h3.png');

    % 清除VLFeat库的路径（如果需要的话）
    rmpath(fullfile(VLFEAT_PATH, 'toolbox'));
end