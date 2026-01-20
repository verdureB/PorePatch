function fpr_at_tpr95 = calculate_fpr95_direct(descriptors, point_ids, pair_indices_matlab)
% CALCULATE_FPR95_DIRECT: 根据描述子、点ID和预定义的描述符对索引计算FPR95。
% 此函数直接翻译自用户提供的Python cal_fpr95逻辑。
%
% 输入参数:
%   descriptors:   DxN 矩阵，D是描述子维度，N是图像块/描述子数量。
%                  每一列是一个描述符。
%   point_ids:     Nx1 或 1xN 向量，包含每个描述符对应的点ID (场景ID或物体ID)。
%                  用于判断哪些是匹配对 (point_id相同) 或非匹配对 (point_id不同)。
%   pair_indices_matlab: Kx2 矩阵，K是预定义描述符对的数量。每一行 [idxA, idxB] 
%                        表示 descriptors(:,idxA) 和 descriptors(:,idxB) 是一对需要评估的描述符。
%                        这些索引必须是1-based的MATLAB索引。
%
% 输出参数:
%   fpr_at_tpr95: 在TPR为95%时的FPR值。

fprintf('开始使用直接翻译的逻辑计算FPR95...\n');

% --- 输入检查 ---
if size(descriptors,2) ~= length(point_ids)
    error('描述符的数量 (列数 %d) 与 point_ids 的数量 (%d) 不匹配。', size(descriptors,2), length(point_ids));
end
if isempty(pair_indices_matlab)
    warning('pair_indices_matlab 为空，无法计算FPR95。');
    fpr_at_tpr95 = NaN;
    return;
end
if min(pair_indices_matlab(:)) < 1 || max(pair_indices_matlab(:)) > size(descriptors,2)
   error('pair_indices_matlab 包含的索引超出了描述符的范围 [1, %d]。请确保它们是1-based且有效。', size(descriptors,2));
end

num_defined_pairs = size(pair_indices_matlab, 1);
all_pair_distances = zeros(num_defined_pairs, 1);
is_positive_pair = false(num_defined_pairs, 1); % 逻辑数组，标记是否为正例对

fprintf('  计算 %d 个预定义描述符对的距离并标记正负例...\n', num_defined_pairs);
for i = 1:num_defined_pairs
    idxA = pair_indices_matlab(i, 1);
    idxB = pair_indices_matlab(i, 2);
    
    descA = descriptors(:, idxA);
    descB = descriptors(:, idxB);
    
    if any(isnan(descA)) || any(isnan(descB))
         % 如果任何一个描述子是NaN，这对的距离也是NaN
         all_pair_distances(i) = NaN;
         % is_positive_pair(i) 保持 false，它们不会被计入有效正负样本
         continue;
    end
    
    all_pair_distances(i) = norm(descA - descB); % L2 距离
    
    % 根据 point_ids 判断是否为正例对
    if point_ids(idxA) == point_ids(idxB)
        is_positive_pair(i) = true;
    else
        is_positive_pair(i) = false;
    end
end

% 移除包含NaN距离的对 (这些对的描述子本身是NaN)
valid_indices = ~isnan(all_pair_distances);
all_pair_distances = all_pair_distances(valid_indices);
is_positive_pair = is_positive_pair(valid_indices);

if isempty(all_pair_distances)
    warning('calculate_fpr95_direct: 没有有效的描述符对用于计算 (所有对都包含NaN描述子)。');
    fpr_at_tpr95 = NaN;
    return;
end

% --- 分离正例和负例的距离 ---
positive_distances = all_pair_distances(is_positive_pair);
negative_distances = all_pair_distances(~is_positive_pair);

num_pos = length(positive_distances);
num_neg = length(negative_distances);

fprintf('    有效正例对数量: %d\n', num_pos);
fprintf('    有效负例对数量: %d\n', num_neg);

if num_pos == 0
    warning('calculate_fpr95_direct: 没有有效的正例对。无法确定TPR95阈值。');
    fpr_at_tpr95 = NaN; % 或者根据情况返回1.0
    return;
end
if num_neg == 0
    warning('calculate_fpr95_direct: 没有有效的负例对。FPR将为0或NaN。');
    % 如果没有负例，任何阈值下的FPR都是0 (如果阈值不导致误判正例)
    % 或者可以认为无法定义FPR。这里返回0。
    fpr_at_tpr95 = 0; 
    return;
end

% --- 找到使TPR为95%的距离阈值 ---
sorted_positive_distances = sort(positive_distances);

% 计算第95百分位的索引
% Python: loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
%         thr = dist_pos[loc_thr-1] (如果loc_thr是1-based长度，或者dist_pos[loc_thr]如果loc_thr是0-based index且长度为numel)
% MATLAB: ceil(num_pos * 0.95) 给出第 k 个元素，使得 k/num_pos >= 0.95
% 例如，num_pos = 20, 0.95*20 = 19. ceil(19) = 19. sorted_positive_distances(19)
% 这意味着19个样本 (95%) 的距离 <= sorted_positive_distances(19)
loc_thr_idx = ceil(num_pos * 0.95);

% 处理边界情况：如果 num_pos 非常小，或者 loc_thr_idx 计算为0
if loc_thr_idx == 0 && num_pos > 0 
    loc_thr_idx = 1; % 至少选择第一个最小距离作为阈值
elseif loc_thr_idx > num_pos % 不应该发生，但作为保护
    loc_thr_idx = num_pos;
end

threshold_at_tpr95 = sorted_positive_distances(loc_thr_idx);
fprintf('  在TPR约为95%%时 (基于第 %d 个正例距离)，距离阈值为: %.6f\n', loc_thr_idx, threshold_at_tpr95);

% --- 利用此阈值计算FPR ---
% FPR = (负例中距离 <= 阈值的数量) / (总负例数量)
num_false_positives = sum(negative_distances <= threshold_at_tpr95);

fpr_at_tpr95 = num_false_positives / num_neg;

fprintf('  在该阈值下，错误接受的负例数量 (False Positives): %d\n', num_false_positives);
fprintf('FPR95 计算完毕。\n');

end