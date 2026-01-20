function descriptor = func_get_patch_descriptor(patch_image, kp_x, kp_y, kp_sigma, kp_orientation)
% FUNC_GET_PATCH_DESCRIPTOR: 为图像块上的单个预定义关键点计算SIFT描述符。
%
% 输入参数:
%   patch_image: 图像块 (例如 64x64, 灰度图, single或double类型)。
%   kp_x, kp_y: 关键点在 patch_image 中的1-based坐标。
%   kp_sigma: 关键点的绝对高斯尺度 (sigma值)。
%   kp_orientation: (可选) 关键点的方向 (弧度)。如果未提供或为空，则默认为0。
%                   注意：你提供的 func_psift.m 可能会将此方向覆盖为0。
%
% 输出参数:
%   descriptor: 计算得到的SIFT描述符 (预期为512x1的列向量)。

% --- 1. 输入参数检查和默认值 ---
if nargin < 4
    error('至少需要提供 patch_image, kp_x, kp_y, kp_sigma 四个参数。');
end
if ~(isnumeric(patch_image) && ismatrix(patch_image) && (size(patch_image,1)==64 && size(patch_image,2)==64))
    error('patch_image 必须是一个64x64的数值矩阵。');
end
if ~(isscalar(kp_x) && isnumeric(kp_x) && kp_x==31 && ...
     isscalar(kp_y) && isnumeric(kp_y) && kp_y==31 && ...
     isscalar(kp_sigma) && isnumeric(kp_sigma) && abs(kp_sigma - sqrt(2)) < 1e-9 && ...
     kp_sigma > 0)
    warning('输入关键点参数与预期的 (31,31,sqrt(2)) 有差异，但仍将继续处理。');
    % 或者，如果严格要求，可以使用 error(...) 替代 warning(...)
end
if nargin < 5 || isempty(kp_orientation)
    kp_orientation_val = 0.0; 
elseif isnumeric(kp_orientation) && isscalar(kp_orientation) && abs(kp_orientation) < 1e-9
    kp_orientation_val = 0.0;
else
    warning('输入方向与预期的0有差异 (%.2f rad)，但仍将继续处理。', kp_orientation);
    kp_orientation_val = kp_orientation; 
end


% 确保图像是single类型且为灰度图
original_patch_image_class = class(patch_image);
patch_image = im2single(patch_image); % SIFT通常在single类型图像上操作
if size(patch_image, 3) > 1
    try
        patch_image = rgb2gray(patch_image);
    catch % 如果没有 Image Processing Toolbox
        if size(patch_image,3) == 3 % 尝试简单平均
             patch_image = mean(patch_image,3);
        else
            error('输入图像 patch_image 是多通道的，但无法转换为灰度图。请提供单通道图像。');
        end
    end
end

fprintf('输入关键点参数: x=%.2f, y=%.2f, sigma=%.4f (sqrt(2)=%.4f), orient=%.2f rad\n', kp_x, kp_y, kp_sigma, sqrt(2), kp_orientation_val);

% --- 2. 为图像块构建高斯金字塔 (GSS) ---
fprintf('正在为图像块构建高斯金字塔 (GSS)...\n');
try
    [~, gss] = func_dog(patch_image);
catch ME
    fprintf('调用 func_dog 时出错: %s\n', ME.message);
    fprintf('请确保 func_dog.m 及其依赖项 (如 gaussianss_lee.m, imsmooth.m) 在MATLAB路径中。\n');
    descriptor = [];
    return;
end

% --- 3. 将全局关键点映射到GSS的倍频程/尺度层级 ---
fprintf('正在将关键点映射到GSS的倍频程/尺度层级...\n');
best_o_sift = -Inf;
best_s_sift = -Inf;
best_o_storage_idx = -1;
min_sigma_diff = inf;

for o_idx = 1:gss.O 
    o_sift_current = gss.omin + o_idx - 1; 
    s_sift_ideal = (log2(kp_sigma) - o_sift_current) * gss.S - 1;
    s_sift_candidate = round(s_sift_ideal);

    if s_sift_candidate >= gss.smin && s_sift_candidate <= gss.smax
        reconstructed_sigma = (2^((s_sift_candidate + 1) / gss.S)) * (2^o_sift_current);
        current_sigma_diff = abs(reconstructed_sigma - kp_sigma);

        % 优先选择sigma差异最小的，如果差异相同，选较低倍频程（较大图像）
        if current_sigma_diff < min_sigma_diff - 1e-6 % 加一点容差避免浮点问题
            min_sigma_diff = current_sigma_diff;
            best_s_sift = s_sift_candidate;
            best_o_sift = o_sift_current;
            best_o_storage_idx = o_idx;
        elseif abs(current_sigma_diff - min_sigma_diff) < 1e-6 
            if best_o_sift > o_sift_current 
                 best_s_sift = s_sift_candidate;
                 best_o_sift = o_sift_current;
                 best_o_storage_idx = o_idx;
            end
        end
    end
end

if best_o_storage_idx == -1
    min_possible_sigma = (2^((gss.smin+1)/gss.S)) * (2^gss.omin);
    max_possible_sigma = (2^((gss.smax+1)/gss.S)) * (2^(gss.omin + gss.O -1));
    warning_msg = sprintf(['无法将 kp_sigma = %f 映射到有效的GSS层级。\n' ...
        '基于gss参数 (omin=%d, O=%d, S=%d, smin=%d, smax=%d, gss.sigma0=%.3f):\n' ...
        '可能的最小sigma约为 %.3f (在oct %d, scale %d)\n' ...
        '可能的最大sigma约为 %.3f (在oct %d, scale %d)\n' ...
        '请检查kp_sigma是否在此范围内，或者图像块对于该kp_sigma是否过小导致倍频程不足。'], ...
        kp_sigma, gss.omin, gss.O, gss.S, gss.smin, gss.smax, gss.sigma0, ...
        min_possible_sigma, gss.omin, gss.smin, ...
        max_possible_sigma, gss.omin + gss.O -1, gss.smax);
    warning(warning_msg);
    descriptor = [];
    return;
end

fprintf('  关键点已映射到: SIFT倍频程 o_sift = %d (GSS存储索引 o_idx = %d), SIFT尺度层级 s_sift = %d\n', ...
        best_o_sift, best_o_storage_idx, best_s_sift);
fprintf('  目标 kp_sigma = %.4f, 重建得到的 sigma = %.4f (绝对差异 = %.3e)\n', ...
        kp_sigma, (2^((best_s_sift+1)/gss.S))*(2^best_o_sift), min_sigma_diff);
% 对于 kp_sigma = sqrt(2), 预期 min_sigma_diff 会非常接近0, o_sift=0, s_sift=3

% --- 4. 为单个关键点准备 'oframes_pr' 和 'ohrs_pr' ---
octave_downscale_factor = 2^best_o_sift;
x_oct = kp_x / octave_downscale_factor;
y_oct = kp_y / octave_downscale_factor;

octave_img_height = size(gss.octave{best_o_storage_idx}, 1);
octave_img_width  = size(gss.octave{best_o_storage_idx}, 2);

if x_oct < 1 || x_oct > octave_img_width || y_oct < 1 || y_oct > octave_img_height
    warning('致命警告: 关键点在目标倍频程图像中的坐标 (%.2f, %.2f) 超出图像边界 (W:%d, H:%d)。无法计算描述符。', x_oct, y_oct, octave_img_width, octave_img_height );
    descriptor = [];
    return;
end

keypoint_octave_data = [x_oct; y_oct; best_s_sift; kp_orientation_val];
oframes_pr_single = cell(1, gss.O); 
for i = 1:gss.O 
    oframes_pr_single{i} = []; 
end
oframes_pr_single{best_o_storage_idx} = keypoint_octave_data;

ohrs_pr_single = cell(1, gss.O);
for i = 1:gss.O
    ohrs_pr_single{i} = [];
end
if ~isempty(oframes_pr_single{best_o_storage_idx}) 
    ohrs_pr_single{best_o_storage_idx} = 1; 
end

% --- 5. 调用 func_psift 计算描述符 ---
fprintf('正在调用 func_psift...\n');
try
    [~, descriptor_output, ~] = func_psift_patch(gss, oframes_pr_single, ohrs_pr_single);
catch ME
    fprintf('调用 func_psift 时出错: %s\n', ME.message);
    fprintf('回溯信息:\n');
    for k=1:length(ME.stack)
        fprintf('文件: %s, 函数: %s, 行: %d\n', ME.stack(k).file, ME.stack(k).name, ME.stack(k).line);
    end
    fprintf('请确保 func_psift.m 及其依赖项 (特别是 siftdescriptor.m) 在MATLAB路径中。\n');
    fprintf('同时检查GSS结构和 oframes_pr 格式是否符合 func_psift 的期望。\n');
    descriptor = [];
    return;
end

% --- 6. 处理并返回描述符 ---
if isempty(descriptor_output)
    fprintf('func_psift 返回了空的描述符。\n');
    descriptor = [];
elseif size(descriptor_output,2) ~=1 && ~isempty(descriptor_output) 
    warning('func_psift 返回了 %d 个描述符，预期为1个。将返回第一个。', size(descriptor_output,2));
    descriptor = descriptor_output(:,1); 
else
    descriptor = descriptor_output;
end

if ~isempty(descriptor)
    fprintf('描述符计算成功 (维度 %d x %d)。\n', size(descriptor,1), size(descriptor,2));
else
    fprintf('未能计算描述符。\n');
end

if ~isempty(descriptor) && ~strcmp(class(descriptor), original_patch_image_class) && strcmp(original_patch_image_class, 'double')
    descriptor = double(descriptor);
end

end