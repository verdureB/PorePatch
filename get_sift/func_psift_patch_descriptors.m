function [d] = func_psift_patch_descriptors(patch)
% date:2025.05.12 name:Li Dong
% patch大小 64*64

VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
f = [31;31;1.5;0];
I_       = vl_imsmooth(im2double(patch), sqrt(f(3)^2 - 0.5^2)) ;
[Ix, Iy] = vl_grad(I_) ;
% 保存Ix和Iy

mod      = sqrt(Ix.^2 + Iy.^2) ;
ang      = atan2(Iy,Ix) ;
grd      = shiftdim(cat(3,mod,ang),2) ;
grd      = single(grd) ;
d        = vl_siftdescriptor(grd, f) ;

% h3的图像，保存
h3 = vl_plotsiftdescriptor(d,f) ;
