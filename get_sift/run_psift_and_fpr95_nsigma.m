% run_sift_and_fpr95_multi_sigma_incremental_save.m
% 主脚本：加载数据，为不同sigma计算SIFT描述子，计算FPR95，并为每个sigma立即保存结果

clear; clc; close all;

% --- 参数设置 ---
python_mat_filename = '/media/human_face_need_test/HyNet/patches_PsPDv4_for_matlab.mat'; % 替换成你实际保存的文件名
output_results_txt_filename = '/media/dataset_maker_matlab_python/get_sift/result/sift_multi_sigma_fpr95_results_incremental_version4.txt'; % 定义保存所有结果的TXT文件名

% --- 定义要测试的sigma值列表 ---
%sigma_values_to_test = [1.0, sqrt(2), 1.8, 2.0, 2.5, 2.0*sqrt(2), 3.5, 4.0]; % 示例sigma值列表
%sigma_values_to_test = [1.0, sqrt(2), 1.8, 2.0, 2.5, 2.0*sqrt(2), 3.5, 4.0]; 
% 你可以根据需要修改这个列表
sigma_values_to_test = [
    3.5000, 
    3.7000
];
fprintf('将测试以下sigma值: %s\n', num2str(sigma_values_to_test));

% --- 0. 初始化结果文件 (写入表头) ---
fprintf('初始化结果文件: %s\n', output_results_txt_filename);
try
    fileID_results = fopen(output_results_txt_filename, 'w'); % 以写入模式打开，清空旧内容
    if fileID_results == -1
        error('无法打开结果文件 "%s" 进行写入。请检查权限或路径。', output_results_txt_filename);
    end
    fprintf(fileID_results, 'Sigma_Value\tFPR95_SIFT\n'); % 写入表头
    fclose(fileID_results); % 关闭文件
    fprintf('结果文件表头已写入。\n');
catch ME_init_save
    fprintf('初始化结果文件时出错: %s\n', ME_init_save.message);
    return; % 如果无法初始化文件，则不继续
end


% --- 1. 加载由Python保存的数据 (只需要加载一次) ---
fprintf('正在加载由Python保存的 .mat 文件: %s...\n', python_mat_filename);
if ~exist(python_mat_filename, 'file')
    error('错误: .mat 文件 "%s" 未找到。', python_mat_filename);
end
try
    loaded_data = load(python_mat_filename);
catch ME
    fprintf('错误：无法加载 .mat 文件: %s\n%s\n', python_mat_filename, ME.message);
    return;
end

required_vars = {'patch_data', 'point_ids', 'match_indices'};
for v = 1:length(required_vars)
    if ~isfield(loaded_data, required_vars{v})
        error('加载的 .mat 文件 "%s" 缺少 "%s"。', python_mat_filename, required_vars{v});
    end
end

python_patches_array = loaded_data.patch_data;
point_ids_from_python = loaded_data.point_ids;
match_indices_from_python = loaded_data.match_indices; 

fprintf('从 .mat 文件加载的变量形状:\n');
fprintf('  patch_data: [%s]\n', num2str(size(python_patches_array)));
fprintf('  point_ids: [%s]\n', num2str(size(point_ids_from_python)));
fprintf('  match_indices: [%s]\n', num2str(size(match_indices_from_python)));

% --- 2. 预处理图像块数据 (只需要进行一次) ---
num_patches = size(python_patches_array, 1); 
num_channels = size(python_patches_array, 2);
patch_height = size(python_patches_array, 3);
patch_width = size(python_patches_array, 4);

if patch_height ~= 64 || patch_width ~= 64
    warning('加载的图像块尺寸不是64x64 (实际为 %dx%d)。', patch_height, patch_width);
end
if num_channels ~= 1
    warning('加载的图像块不是单通道灰度图 (实际通道数: %d)。', num_channels);
end

fprintf('将 %d 个图像块 (尺寸 %dx%dx%d) 转换为MATLAB cell数组格式...\n', num_patches, num_channels, patch_height, patch_width);
matlab_patches_cell = cell(num_patches, 1);
for i = 1:num_patches
    matlab_patches_cell{i} = squeeze(python_patches_array(i, :, :, :)); 
    if size(matlab_patches_cell{i},3) > 1 && num_channels > 1 
        matlab_patches_cell{i} = matlab_patches_cell{i}(:,:,1);
    end
end
fprintf('图像块转换完成。\n');

% --- 准备存储所有sigma的FPR95结果 (可选，主要用于最终概览或调试) ---
results_in_memory = []; % 用于在内存中存储 [sigma, fpr95] 对

% --- 3. 循环处理不同的sigma值 ---
for sigma_idx = 1:length(sigma_values_to_test)
    current_sigma_val = sigma_values_to_test(sigma_idx);
    fprintf('\n=====================================================\n');
    fprintf('开始处理 SIGMA = %.4f (第 %d / %d 个sigma值)\n', current_sigma_val, sigma_idx, length(sigma_values_to_test));
    fprintf('=====================================================\n');

    % --- 3a. 计算SIFT描述子 ---
    fprintf('  正在为所有图像块计算SIFT描述子 (sigma=%.4f)...\n', current_sigma_val);
    all_sift_descriptors_current_sigma = []; 
    try
        % 调用新的批处理函数，传入当前的sigma值
        all_sift_descriptors_current_sigma = func_process_multiple_patches_nsigma(matlab_patches_cell, current_sigma_val);
    catch ME
        fprintf('  为 sigma=%.4f 计算SIFT描述子时出错: %s\n', current_sigma_val, ME.message);
        fprintf('  错误详情:\n');
        for k_err=1:length(ME.stack)
            fprintf('    文件: %s, 函数: %s, 行: %d\n', ME.stack(k_err).file, ME.stack(k_err).name, ME.stack(k_err).line);
        end
        results_in_memory = [results_in_memory; [current_sigma_val, NaN]]; % 记录失败到内存
        % 将NaN结果也写入文件
        try
            fileID_results = fopen(output_results_txt_filename, 'a'); % 以追加模式打开文件
            if fileID_results == -1
                error('无法打开结果文件 "%s" 进行追加写入。', output_results_txt_filename);
            end
            fprintf(fileID_results, '%.4f\t\t%.8f\n', current_sigma_val, NaN);
            fclose(fileID_results);
        catch ME_save_err
            fprintf('  为 sigma=%.4f 保存NaN到TXT时出错: %s\n', current_sigma_val, ME_save_err.message);
        end
        continue; % 跳到下一个sigma值
    end

    if isempty(all_sift_descriptors_current_sigma)
        fprintf('  错误：为 sigma=%.4f 未能计算任何SIFT描述子。\n', current_sigma_val);
        results_in_memory = [results_in_memory; [current_sigma_val, NaN]];
        try
            fileID_results = fopen(output_results_txt_filename, 'a'); 
            fprintf(fileID_results, '%.4f\t\t%.8f\n', current_sigma_val, NaN);
            fclose(fileID_results);
        catch ME_save_err
             fprintf('  为 sigma=%.4f 保存NaN到TXT时出错: %s\n', current_sigma_val, ME_save_err.message);
        end
        continue;
    end
    if any(all(isnan(all_sift_descriptors_current_sigma), 1))
        nan_cols = find(all(isnan(all_sift_descriptors_current_sigma), 1));
        fprintf('  警告：为 sigma=%.4f，以下 %d 个图像块未能成功计算描述子: %s\n', current_sigma_val, length(nan_cols), num2str(nan_cols));
    end
    fprintf('  SIFT描述子计算完成 (sigma=%.4f)。描述子矩阵大小: [%s]\n', current_sigma_val, num2str(size(all_sift_descriptors_current_sigma)));

    % --- 3b. 计算FPR95 ---
    fpr95_current_sigma = NaN; 
    if min(match_indices_from_python(:)) == 0
        pair_indices_matlab = match_indices_from_python + 1;
    else
        pair_indices_matlab = match_indices_from_python;
    end

    temp_point_ids = point_ids_from_python; 
    if size(temp_point_ids, 2) > size(temp_point_ids, 1) && size(temp_point_ids, 1) == 1
        temp_point_ids = temp_point_ids'; 
    elseif ~(size(temp_point_ids, 1) > size(temp_point_ids, 2) && size(temp_point_ids, 2) == 1) 
        if length(temp_point_ids) == size(all_sift_descriptors_current_sigma,2) && numel(temp_point_ids) == length(temp_point_ids)
             temp_point_ids = reshape(temp_point_ids, [], 1); 
        else
            warning('  point_ids 的形状 [%s] 可能不适合 (sigma=%.4f)。', num2str(size(temp_point_ids)), current_sigma_val);
        end
    end

    if ~isempty(all_sift_descriptors_current_sigma) && ~all(isnan(all_sift_descriptors_current_sigma(:)), 'all')
        fprintf('  准备调用 calculate_fpr95_direct (sigma=%.4f)...\n', current_sigma_val);
        try
            fpr95_current_sigma = calculate_fpr95_direct(all_sift_descriptors_current_sigma, temp_point_ids, pair_indices_matlab);
            fprintf('  -------------------------------------------\n');
            fprintf('  计算得到的 SIFT FPR95 (sigma=%.4f): %.8f\n', current_sigma_val, fpr95_current_sigma);
            fprintf('  -------------------------------------------\n');
        catch ME_fpr
            fprintf('  为 sigma=%.4f 计算FPR95时出错: %s\n', current_sigma_val, ME_fpr.message);
        end
    else
        fprintf('  没有有效的SIFT描述子可用于为 sigma=%.4f 计算FPR95。\n', current_sigma_val);
    end
    results_in_memory = [results_in_memory; [current_sigma_val, fpr95_current_sigma]]; % 仍然可以保存在内存中以供最后查看
    
    % --- 3c. 将当前sigma的FPR95结果追加到TXT文件 ---
    fprintf('  正在将 sigma=%.4f 的FPR95结果追加到文件: %s...\n', current_sigma_val, output_results_txt_filename);
    try
        fileID_results = fopen(output_results_txt_filename, 'a'); % 以追加模式 ('a') 打开文件
        if fileID_results == -1
            error('无法打开结果文件 "%s" 进行追加写入。', output_results_txt_filename);
        end
        fprintf(fileID_results, '%.4f\t\t%.8f\n', current_sigma_val, fpr95_current_sigma); % 写入当前sigma和FPR95
        fclose(fileID_results);
        fprintf('  sigma=%.4f 的FPR95结果已成功追加。\n', current_sigma_val);
    catch ME_append_save
        fprintf('  为 sigma=%.4f 追加FPR95结果到TXT文件时出错: %s\n', current_sigma_val, ME_append_save.message);
    end
    
end % 结束对所有sigma值的循环

% --- 4. (可选) 最终确认所有结果已写入 ---
% 上一步已经在循环中写入了，这一步可以移除或保留用于显示内存中的结果
fprintf('\n=====================================================\n');
fprintf('所有sigma值处理完毕。结果已增量保存到: %s\n', output_results_txt_filename);
if ~isempty(results_in_memory)
    fprintf('内存中收集的结果 (Sigma, FPR95):\n');
    disp(results_in_memory);
end
fprintf('主脚本执行完毕。\n');