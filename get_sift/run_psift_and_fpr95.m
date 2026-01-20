% run_sift_and_fpr95.m
% 主脚本：加载数据，计算SIFT描述子，计算FPR95，并保存FPR95结果

clear; clc; close all;

% --- 参数设置 ---
python_mat_filename = '/media/human_face_need_test/HyNet/patches_PsPDv3_for_matlab.mat'; % 替换成你实际保存的文件名
output_fpr_txt_filename = 'sift_fpr95_result.txt'; % 定义保存FPR95结果的TXT文件名

% --- 1. 加载由Python保存的数据 ---
fprintf('正在加载由Python保存的 .mat 文件: %s...\n', python_mat_filename);
if ~exist(python_mat_filename, 'file')
    error('错误: .mat 文件 "%s" 未找到。请确保Python脚本已成功保存该文件，并且路径正确。', python_mat_filename);
end
try
    loaded_data = load(python_mat_filename);
catch ME
    fprintf('错误：无法加载 .mat 文件: %s\n', python_mat_filename);
    fprintf('%s\n', ME.message);
    return;
end

% 检查加载的变量是否存在
required_vars = {'patch_data', 'point_ids', 'match_indices'};
for v = 1:length(required_vars)
    if ~isfield(loaded_data, required_vars{v})
        error('加载的 .mat 文件 "%s" 缺少必要的变量 "%s"。', python_mat_filename, required_vars{v});
    end
end

python_patches_array = loaded_data.patch_data;
point_ids_from_python = loaded_data.point_ids;
match_indices_from_python = loaded_data.match_indices; 

fprintf('从 .mat 文件加载的变量形状:\n');
fprintf('  patch_data: [%s]\n', num2str(size(python_patches_array)));
fprintf('  point_ids: [%s]\n', num2str(size(point_ids_from_python)));
fprintf('  match_indices: [%s]\n', num2str(size(match_indices_from_python)));

% --- 2. 预处理图像块数据 ---
num_patches = size(python_patches_array, 1); 
num_channels = size(python_patches_array, 2);
patch_height = size(python_patches_array, 3);
patch_width = size(python_patches_array, 4);

if patch_height ~= 64 || patch_width ~= 64
    warning('加载的图像块尺寸不是64x64 (实际为 %dx%d)。后续SIFT处理可能不符合预期。', patch_height, patch_width);
end
if num_channels ~= 1
    warning('加载的图像块不是单通道灰度图 (实际通道数: %d)。SIFT通常处理灰度图。squeeze操作将移除此通道。', num_channels);
end

fprintf('将 %d 个图像块 (尺寸 %dx%dx%d) 转换为MATLAB cell数组格式...\n', num_patches, num_channels, patch_height, patch_width);
matlab_patches_cell = cell(num_patches, 1);
for i = 1:num_patches
    matlab_patches_cell{i} = squeeze(python_patches_array(i, :, :, :)); 
    if size(matlab_patches_cell{i},3) > 1 && num_channels > 1 
        warning('图像块 %d 在squeeze后仍然是多维的，可能不是灰度图。取第一个通道。', i);
        matlab_patches_cell{i} = matlab_patches_cell{i}(:,:,1);
    end
end
fprintf('图像块转换完成。\n');

% --- 3. 计算SIFT描述子 ---
fprintf('正在为所有图像块计算SIFT描述子...\n');
all_sift_descriptors = []; 
try
    all_sift_descriptors = func_process_multiple_patches(matlab_patches_cell);
catch ME
    fprintf('计算SIFT描述子时出错: %s\n', ME.message);
    fprintf('请确保所有SIFT相关的 .m 文件都在MATLAB路径中且功能正常。\n');
    fprintf('错误详情:\n');
    for k_err=1:length(ME.stack)
        fprintf('  文件: %s, 函数: %s, 行: %d\n', ME.stack(k_err).file, ME.stack(k_err).name, ME.stack(k_err).line);
    end
    return; 
end

if isempty(all_sift_descriptors)
    fprintf('错误：未能计算任何SIFT描述子。\n');
    return;
end
if any(all(isnan(all_sift_descriptors), 1))
    nan_cols = find(all(isnan(all_sift_descriptors), 1));
    fprintf('警告：以下 %d 个图像块未能成功计算描述子 (其列为NaN): %s\n', length(nan_cols), num2str(nan_cols));
end
fprintf('SIFT描述子计算完成。描述子矩阵大小: [%s]\n', num2str(size(all_sift_descriptors)));

% --- 4. 计算FPR95 ---
fpr95_sift = NaN; % 初始化FPR95值为NaN
if min(match_indices_from_python(:)) == 0
    fprintf('match_indices 似乎是0-based，正在转换为1-based...\n');
    pair_indices_matlab = match_indices_from_python + 1;
else
    fprintf('match_indices 似乎已经是1-based (最小值为 %d)。\n', min(match_indices_from_python(:)));
    pair_indices_matlab = match_indices_from_python;
end

if size(point_ids_from_python, 2) > size(point_ids_from_python, 1) && size(point_ids_from_python, 1) == 1
    point_ids_from_python = point_ids_from_python'; 
elseif size(point_ids_from_python, 1) > size(point_ids_from_python, 2) && size(point_ids_from_python, 2) == 1
    % 已经是列向量
else
    if length(point_ids_from_python) == size(all_sift_descriptors,2) && numel(point_ids_from_python) == length(point_ids_from_python)
         point_ids_from_python = reshape(point_ids_from_python, [], 1); 
         fprintf('point_ids 被重塑为列向量。\n');
    else
        warning('point_ids 的形状 [%s] 可能不适合，预期为 N x 1 列向量。', num2str(size(point_ids_from_python)));
    end
end

if ~isempty(all_sift_descriptors) && ~all(isnan(all_sift_descriptors(:)), 'all')
    fprintf('准备调用 calculate_fpr95_direct...\n');
    try
        fpr95_sift = calculate_fpr95_direct(all_sift_descriptors, point_ids_from_python, pair_indices_matlab);
        fprintf('-------------------------------------------\n');
        fprintf('计算得到的 SIFT FPR95: %.8f\n', fpr95_sift); % 增加小数点位数以便更精确显示
        fprintf('-------------------------------------------\n');
    catch ME_fpr
        fprintf('计算FPR95时出错: %s\n', ME_fpr.message);
        fprintf('FPR95计算函数的堆栈信息:\n');
        for k_fpr=1:length(ME_fpr.stack)
            fprintf('  文件: %s, 函数: %s, 行: %d\n', ME_fpr.stack(k_fpr).file, ME_fpr.stack(k_fpr).name, ME_fpr.stack(k_fpr).line);
        end
    end
else
    fprintf('没有有效的SIFT描述子可用于计算FPR95。\n');
end

% --- 5. 将FPR95结果保存到TXT文件 ---
if ~isnan(fpr95_sift)
    fprintf('正在将FPR95结果保存到文件: %s...\n', output_fpr_txt_filename);
    try
        fileID = fopen(output_fpr_txt_filename, 'w'); % 以写入模式打开文件，如果文件已存在则覆盖
        if fileID == -1
            error('无法打开文件 "%s" 进行写入。请检查权限或路径。', output_fpr_txt_filename);
        end
        fprintf(fileID, 'SIFT_FPR95: %.8f\n', fpr95_sift); % 将FPR95值写入文件，保留8位小数
        fclose(fileID);
        fprintf('FPR95结果已成功保存。\n');
    catch ME_save
        fprintf('保存FPR95结果到TXT文件时出错: %s\n', ME_save.message);
    end
else
    fprintf('FPR95值为NaN，未保存到TXT文件。\n');
end

fprintf('主脚本执行完毕。\n');