function [frames,descriptors,hrs_out] = func_psift_patch(gss,oframes_pr,ohrs_pr) % Renamed hrs to hrs_out to avoid conflict
%UNTITLED ˴ʾйش˺ժҪ
%   ˴ʾϸ˵
mmm=2;
magnif = 3.0 ;
NBP=8;
NBO=8;
verb=1;
frames      = [] ;
hrs_out     = []; % Initialize output hrs
descriptors = [] ;

    for o=1:gss.O % o 是 gss.octave 和 oframes_pr 的存储索引 (1-based)
        current_oframes = oframes_pr{o}; % 当前倍频程的帧数据
        current_ohrs    = ohrs_pr{o};    % 当前倍频程的Hessian数据

        if ~isempty(current_oframes) % !!核心修改：只在当前倍频程有关键点数据时处理!!
            
            % 确保 current_oframes 至少有3行 (x, y, s)
            if size(current_oframes, 1) < 3
                error('func_psift: oframes_pr{%d} 的行数少于3，数据不完整。', o);
            end

            % 方向处理：原始代码是无条件设置方向为0。
            % 如果 current_oframes 传入时已经有4行（即包含了方向），
            % 这里的代码将覆盖它。如果希望使用传入的方向，此逻辑需调整。
            % 对于你的固定输入(方向为0)，当前逻辑结果一致。
            temp_frames_with_orientation = current_oframes; % 先复制
            if size(temp_frames_with_orientation,1) < 4
                 temp_frames_with_orientation(4, :) = 0; % 如果没有第4行，添加并设为0
            else
                 temp_frames_with_orientation(4, :) = 0; % 如果有第4行，也将其覆盖为0 (遵循原始逻辑)
            end
            
            % oframes(3,:)=oframes(3,:); % 原始代码中的这行是多余的

            % 转换到全局坐标系并存储 frames
            x     = 2^(o-1+gss.omin) * temp_frames_with_orientation(1,:) ;
            y     = 2^(o-1+gss.omin) * temp_frames_with_orientation(2,:) ;
            sigma = 2^(o-1+gss.omin) * gss.sigma0 * 2.^(temp_frames_with_orientation(3,:)/gss.S) ;
            
            frames = [frames, [x(:)' ; y(:)' ; sigma(:)' ; temp_frames_with_orientation(4,:)] ];
            
            % 处理 Hessian 响应 (hrs)
            if ~isempty(current_ohrs)
                % 假设 current_ohrs 是一个行向量，或者我们只关心它的第一行
                % （如果它是单个标量值 1，current_ohrs(1,:) 也是 1）
                if size(current_ohrs, 1) > 0 
                    hrs_out =[hrs_out, current_ohrs(1,:)];
                end
            end
            
            % 计算描述符
            if nargout > 1 % 如果调用者请求了 descriptors 输出
                if verb > 0
                    fprintf('PSIFT: computing descriptors at oct %d (SIFT oct %d)...',o, o-1+gss.omin) ; % 显示存储索引和SIFT倍频程索引
                    tic ;
                end
                
                image_data_for_descriptor = gss.octave{o}; % 使用当前倍频程的高斯平滑图像
                
                % 调用 siftdescriptor
                sh = siftdescriptor(...
                        image_data_for_descriptor, ...
                        temp_frames_with_orientation, ... % 包含4行 [x_oct;y_oct;s_oct;orientation_oct]
                        gss.sigma0, ...                   % 即 2^(1/gss.S)
                        gss.S, ...
                        gss.smin, ...
                        'Magnif',mmm* magnif, ...         % 有效 Magnif = 6.0
                        'NumSpatialBins', NBP, ...        % 8
                        'NumOrientBins', NBO);            % 8
                
                descriptors = [descriptors, sh] ;
                
                if verb > 0, fprintf('done (%.3f s)\n',toc) ; end
            end
        else
            % 如果 current_oframes 为空，可以选择打印一条跳过信息
            if verb > 0
                % fprintf('PSIFT: Skipping empty octave storage index %d\n', o);
            end
        end % 结束 if ~isempty(current_oframes)
    end % 结束 for o=1:gss.O
end