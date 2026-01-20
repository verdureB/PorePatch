%先读入关键点文件
clc
clear
close all

%vl库的路径所在
VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
file_content = fileread('/media/human_face_need_test/brown_v2/transfromer_bin_to_point.txt');

% 按行分割
lines = strsplit(file_content, '\n');
%读入获取参数
% 初始化参数
data_path = '';
human = '';
front_cam = '';
select_frames = '';
emotion = 'EMO-1-shout+laugh';
% 解析每一行
for i = 1:length(lines)
    line = lines{i};
    if contains(line, '=')
        % 按等号分割键值对
        parts = strsplit(line, '=');
        key = strtrim(parts{1});   % 去除空格
        value = strtrim(parts{2}); % 去除空格
        
        % 根据键赋值
        switch key
            case 'data_path'
                data_path = value;
            case 'human'
                human = value;
            case 'front_cam'
                front_cam = value; % 转换为数值
            case 'select_frames'
                select_frames = value;
            otherwise
                warning('未知参数: %s', key);
        end
    end
end



cam = front_cam;
people_num = human;
%读取bin文件转换成keypoints
path = sprintf('%s/%s/%s/%s/psiftproject/keypoints/%s.bmp.bin',data_path,people_num,emotion,select_frames,cam);
disp(path)
point = read_keypoints(path)';
px=point(1,:);
py=point(2,:);
px=round(px);
py=round(py);
point=[px;py];
%存成yml文件和后面的对接
output_path_yml=sprintf('%s/%s/%s/%s/%s_front_kpt.yml',data_path,people_num,emotion,select_frames,cam);
dym_matlab2opencv(point,output_path_yml)