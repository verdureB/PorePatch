function [d] = func_psift_patch_descriptors2(patch)
    % date: 2025.05.12 name: Li Dong
    % patch大小 64*64

    % 设置VLFeat库的路径并初始化
    VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
    addpath(fullfile(VLFEAT_PATH, 'toolbox'));
    vl_setup;

    
    
    
    patch = 
    figure;
    image(patch); % 显示patch图像
    hold on;
    h1 = vl_plotframe(f);
    h2 = vl_plotframe(f);
    set(h1, 'color', 'k', 'linewidth', 3);
    set(h2, 'color', 'y', 'linewidth', 2);
    h3 = vl_plotsiftdescriptor(d, f);
    set(h3, 'color', 'r'); % 设置描述子的颜色为绿色
    hold off;

    % 保存图像
    frame = getframe(gca);
    im = frame.cdata;
    [imind, cm] = rgb2ind(im, 256);
    imwrite(imind, cm, '/media/dataset_maker_matlab_python/get_sift/aimages/h3.png');
end