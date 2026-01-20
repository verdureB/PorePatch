function all_descriptors = func_process_multiple_patches_save_png(patch_images_cell)
% FUNC_PROCESS_MULTIPLE_PATCHES: 为一批图像块计算SIFT描述符。
% 对每个图像块使用固定的关键点参数 (x=31, y=31, sigma=sqrt(2), orientation=0)。
%
% 输入参数:
%   patch_images_cell: 一个 Nx1 的单元数组，其中每个单元 patch_images_cell{i}
%                      包含一个64x64的图像块 (灰度图, single或double类型)。
%
% 输出参数:
%   all_descriptors: 一个 DxN 的矩阵，其中 D 是描述符的长度 (根据你的设置为512)，
%                    N 是输入图像块的数量。每一列对应一个图像块的描述符。
%                    如果某个图像块的描述符未能计算，其对应的列将是 NaNs。

% --- 1. 固定的关键点参数 (根据你的指定) ---
kp_x_fixed = 31;
kp_y_fixed = 31;
%kp_sigma_fixed = sqrt(2);% origin 为sqrt（2） lin改成2-----2025.5.14
kp_sigma_fixed = 2
kp_orientation_fixed = 0;
descriptor_length = 512; % 基于 func_psift_patch 中 NBP=8, NBO=8
fprintf('开始处理图像块...\n');
% --- 2. 输入验证 ---
if ~iscell(patch_images_cell) || isempty(patch_images_cell)
    error('输入必须是一个非空的图像块单元数组。');
end
num_patches = length(patch_images_cell);
% 用NaN预分配输出矩阵，方便识别处理失败的patch
all_descriptors = NaN(descriptor_length, num_patches, 'single'); 

fprintf('开始处理 %d 个图像块...\n', num_patches);

% --- 3. 循环处理每个图像块 ---
for i = 1:num_patches
    fprintf('正在处理第 %d / %d 个图像块...\n', i, num_patches);
    patch_image = patch_images_cell{i};

    % --- 3a. 图像块验证和预处理 (源自处理单个patch的函数) ---
    if ~(isnumeric(patch_image) && ismatrix(patch_image) && ...
         (size(patch_image,1)==64 && size(patch_image,2)==64))
        warning('图像块 %d 不是一个64x64的数值矩阵。正在跳过此图像块。', i);
        continue; % 跳过当前图像块，处理下一个
    end
    
    patch_image_single = im2single(patch_image); % SIFT通常在single类型图像上操作
    if size(patch_image_single, 3) > 1
        try
            patch_image_single = rgb2gray(patch_image_single);
        catch % 如果没有 Image Processing Toolbox
            if size(patch_image_single,3) == 3 % 尝试简单平均
                 patch_image_single = mean(patch_image_single,3);
            else
                warning('图像块 %d 是多通道图像且无法转换为灰度图。正在跳过此图像块。', i);
                continue;
            end
        end
    end

    % 使用固定的关键点参数
    current_kp_x = kp_x_fixed;
    current_kp_y = kp_y_fixed;
    current_kp_sigma = kp_sigma_fixed;
    current_kp_orientation = kp_orientation_fixed;

    fprintf('  图像块 %d 使用关键点: x=%.2f, y=%.2f, sigma=%.4f, orient=%.2f rad\n', ...
        i, current_kp_x, current_kp_y, current_kp_sigma, current_kp_orientation);

    % --- 3b. 为当前图像块构建高斯金字塔 (GSS) ---
    gss = []; % 确保每次循环的gss是新的
    try
        % 注意：func_dog 是你提供的原始文件之一
        [dogss, gss] = func_dog(patch_image_single);
    catch ME
        warning('为图像块 %d 构建GSS时出错: %s。正在跳过此图像块。', i, ME.message);
        continue;
    end

    % --- 3c. 将关键点映射到GSS的倍频程/尺度层级 ---
    best_o_sift = -Inf;
    best_s_sift = -Inf;
    best_o_storage_idx = -1;
    min_sigma_diff = inf;
    %gss.O  octave组数
    for o_idx = 1:gss.O 
    %目的: 这一行的作用是将基于1的存储索引 o_idx 转换为在SIFT理论中通用的、基于0的SIFT倍频程索引
        o_sift_current = gss.omin + o_idx - 1; 
        %计算出，如果是这个octave的，对应的应该是那个layer
        s_sift_ideal = (log2(current_kp_sigma) - o_sift_current) * gss.S - 1;
        s_sift_candidate = round(s_sift_ideal);
        %在正常可选的layer中
        if s_sift_candidate >= gss.smin && s_sift_candidate <= gss.smax
            reconstructed_sigma = (2^((s_sift_candidate + 1) / gss.S)) * (2^o_sift_current);
            current_sigma_diff = abs(reconstructed_sigma - current_kp_sigma);
            %计算对应算到的layer的scale 和设定好的scale谁更接近，更接近的作为其的octave
            if current_sigma_diff < min_sigma_diff - 1e-6 
                min_sigma_diff = current_sigma_diff;
                best_s_sift = s_sift_candidate;
                best_o_sift = o_sift_current;
                best_o_storage_idx = o_idx;
            elseif abs(current_sigma_diff - min_sigma_diff) < 1e-6 
                %如果两个被选中的  o和s计算出来的scale都和sqrt(2)接近，我们就使用更小的O
                if best_o_sift > o_sift_current 
                     best_s_sift = s_sift_candidate;
                     best_o_sift = o_sift_current;
                     best_o_storage_idx = o_idx;
                end
            end
        end
    end
%    




% --- 在 "3c. 将关键点映射到GSS的倍频程/尺度层级 ---" 之后添加 ---
% --- 用于保存最佳匹配层图像的核心代码 (使用 mat2gray) ---

if best_o_storage_idx ~= -1 % 确保在之前的映射步骤中找到了有效的匹配层
    
    % 计算该层在 gss.octave{best_o_storage_idx} 的第三个维度上的1-based存储索引
    s_storage_idx_in_octave = best_s_sift - gss.smin + 1;

    % 边界检查，确保索引有效
    if best_o_storage_idx >= 1 && best_o_storage_idx <= length(gss.octave) && ...
       s_storage_idx_in_octave >= 1 && ...
       s_storage_idx_in_octave <= size(gss.octave{best_o_storage_idx}, 3)
        
        selected_layer_image = gss.octave{best_o_storage_idx}(:, :, s_storage_idx_in_octave);
        selected_layer_image_dog = dogss.octave{best_o_storage_idx}(:, :, s_storage_idx_in_octave);
        % --- 调试信息：输出原始图像层的值范围 ---
        min_val_raw = min(selected_layer_image(:));
        max_val_raw = max(selected_layer_image(:));
        fprintf('    原始 selected_layer_image 值范围: [min=%.4f, max=%.4f]\n', min_val_raw, max_val_raw);
        % ------------------------------------

        % 使用 mat2gray 将图像数据归一化到 [0, 1] 范围
        % 这会自动处理 min_val_raw 和 max_val_raw，将它们分别映射到 0 和 1
        % 如果图像是纯色的 (min_val_raw == max_val_raw)，mat2gray 会返回一个全0的图像
        % （除非原始纯色值非常大，具体行为可查阅mat2gray文档，但通常目标是可视化对比度）
    
        img_to_save_normalized = mat2gray(selected_layer_image);
        img_to_save_normalized_dog =mat2gray(selected_layer_image_dog);
        % --- 调试信息：输出归一化后图像层的值范围 ---
        % min_val_norm = min(img_to_save_normalized(:));
        % max_val_norm = max(img_to_save_normalized(:));
        % fprintf('    mat2gray 处理后图像值范围: [min=%.4f, max=%.4f]\n', min_val_norm, max_val_norm);
        % ------------------------------------
        sizea = size(img_to_save_normalized,1)
        sizeb = size(img_to_save_normalized,2)
        % 定义输出文件名
        output_image_filename = sprintf('%d_%d_gss_layer_oct%d_s%d_sigma_target%.2f.png', ...
                                        sizea,sizeb,best_o_sift, best_s_sift, current_kp_sigma);
        


        output_image_filename_dog = sprintf('%d_%d_dogss_layer_oct%d_s%d_sigma_target%.2f.png', ...
                                    sizea,sizeb,best_o_sift, best_s_sift, current_kp_sigma);
        % 尝试保存图像
        try
            imwrite(img_to_save_normalized, output_image_filename);
            imwrite(img_to_save_normalized_dog, output_image_filename_dog); % 保存归一化后的图像
            fprintf('    最佳匹配层图像已保存为: %s\n', output_image_filename);

            % --- 可选：同时显示一下这张即将保存的图像 ---
            % figure;
            % imshow(img_to_save_normalized); % mat2gray 的输出可以直接给 imshow
            % title(['预览保存的图像 (mat2gray): ' strrep(output_image_filename, '_', '\_')]);
            % ------------------------------------------------
        catch ME_imwrite
            fprintf('    保存图像 "%s" 时出错: %s\n', output_image_filename, ME_imwrite.message);
        end
        
    else
        fprintf('    错误: 计算得到的最佳层索引 (oct_storage_idx=%d, scale_storage_idx=%d) 超出gss.octave的范围。\n', ...
                best_o_storage_idx, s_storage_idx_in_octave);
    end
else
    % fprintf('  未能找到关键点对应的有效高斯金字塔层级，无法保存图像。\n');
end
% --- 结束图像保存代码 ---








    if best_o_storage_idx == -1
        warning('无法为图像块 %d 将关键点 sigma=%.4f 映射到有效的GSS层级。正在跳过此图像块。', i, current_kp_sigma);
        continue;
    end
    fprintf('    图像块 %d 的关键点已映射到: SIFT倍频程 o_sift = %d (GSS存储索引 o_idx = %d), SIFT尺度层级 s_sift = %d\n', ...
        i, best_o_sift, best_o_storage_idx, best_s_sift);

    % --- 3d. 为单个关键点准备 'oframes_pr' 和 'ohrs_pr' ---
    octave_downscale_factor = 2^best_o_sift;
    x_oct = current_kp_x / octave_downscale_factor;
    y_oct = current_kp_y / octave_downscale_factor;

    octave_img_height = size(gss.octave{best_o_storage_idx}, 1);
    octave_img_width  = size(gss.octave{best_o_storage_idx}, 2);

    if x_oct < 1 || x_oct > octave_img_width || y_oct < 1 || y_oct > octave_img_height
        warning('图像块 %d 的关键点坐标 (%.2f, %.2f) 超出其目标倍频程图像边界 (W:%d, H:%d)。正在跳过此图像块。', ...
                i, x_oct, y_oct, octave_img_width, octave_img_height );
        continue;
    end

    keypoint_octave_data = [x_oct; y_oct; best_s_sift; current_kp_orientation];
    oframes_pr_single = cell(1, gss.O); 
    for k_cell = 1:gss.O; oframes_pr_single{k_cell} = []; end % 初始化为空
    oframes_pr_single{best_o_storage_idx} = keypoint_octave_data;

    ohrs_pr_single = cell(1, gss.O);
    for k_cell = 1:gss.O; ohrs_pr_single{k_cell} = []; end % 初始化为空
    if ~isempty(oframes_pr_single{best_o_storage_idx}) 
        ohrs_pr_single{best_o_storage_idx} = 1; 
    end

    % --- 3e. 调用 func_psift_patch 计算描述符 ---
    % 假设你已经将 func_psift.m 重命名（或其内部函数名修改）为 func_psift_patch.m
    descriptor_output_current_patch = []; % 为当前patch初始化
    try
        [~, descriptor_output_current_patch, ~] = func_psift_patch(gss, oframes_pr_single, ohrs_pr_single);
    catch ME
        warning('为图像块 %d 调用 func_psift_patch 时出错: %s。正在跳过此图像块。', i, ME.message);
        % 可以在这里添加更详细的错误信息打印，如 ME.stack
        continue;
    end

    % --- 3f. 存储当前图像块的描述符 ---
    if isempty(descriptor_output_current_patch)
        warning('func_psift_patch 为图像块 %d 返回了空的描述符。', i);
    elseif size(descriptor_output_current_patch, 2) ~= 1
        warning('func_psift_patch 为图像块 %d 返回了 %d 个描述符 (预期为1个)。将使用第一个。', i, size(descriptor_output_current_patch,2));
        all_descriptors(:, i) = single(descriptor_output_current_patch(:,1)); % 确保存储为single
    else
        all_descriptors(:, i) = single(descriptor_output_current_patch); % 确保存储为single
        fprintf('    图像块 %d 的描述符计算成功。\n', i);
    end
    
end % 结束对所有图像块的循环

fprintf('所有 %d 个图像块处理完毕。\n', num_patches);

end