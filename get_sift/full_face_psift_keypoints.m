clc
clear
close all

%vl库的路径所在
VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
file_content = fileread('/media/human_face_need_test/brown_v2/get_sift_parameters.txt');

% 按行分割
lines = strsplit(file_content, '\n');
%读入获取参数
% 初始化参数
py_path = '';
py_people = '';
no_kpt_py = 0;

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
            case 'dataset_path_call_matlab'
                py_path = value;
            case 'human_number'
                py_people = value;
            case 'kpts_number'
                no_kpt_py = str2double(value); % 转换为数值
            otherwise
                warning('未知参数: %s', key);
        end
    end
end



% 输出参数
%fprintf('dataset_path: %s\n', py_path);
%fprintf('human_number: %s\n', py_people);
%fprintf('kpts_number: %d\n', no_kpt_py);






dist_eyes=1/1.6;
no_kpt=no_kpt_py;
%emotions={"disgust","fear","sadness","surprise","netural","happy"};
emotions={"EMO-1-shout+laugh"};
%people={"001","002","003","004","005","006","007","009","008","010","011"};
path = py_path;
pre_path = py_path;

folders = dir(path);



%people ={"017"};
people = {py_people};
for e =1:1
    emotion=emotions{e};
    for p_num=1:length(people)
    people_num=people{p_num};
    
        for cam = 1:16

            for frame=10:10
            %读入图片
            face_path = sprintf('%s/%s/%s/%d/psiftproject/images/%d.bmp', pre_path,people_num,emotion,frame,cam);
            image_face = imread(face_path); 
            
            %转换成灰度图便于后续的keypoints检测
            image_face = rgb2gray(image_face);      

            %读入mask
            mask_path = sprintf('%s/%s/%s/%d/psiftproject/mask/%d_mask.bmp', pre_path,people_num,emotion,frame,cam);
            image_mask = imread(mask_path);  

        

            output_dir = sprintf('%s/%s/%s/%d/psiftproject/keypoints', pre_path,people_num,emotion,frame);
            output_psfit_descriptors_dir = sprintf('%s/%s/%s/%d/psiftproject/descriptors', pre_path,people_num,emotion,frame);
            if exist(output_dir, 'dir') ~= 7
                mkdir(output_dir);
                disp(['Folder created: ', output_dir]);
            else
                %disp(['Folder already exists: ', output_dir]);
            end

            if exist(output_psfit_descriptors_dir, 'dir') ~= 7
                mkdir(output_psfit_descriptors_dir);
                disp(['Folder created: ', output_psfit_descriptors_dir]);
            else
                %disp(['Folder already exists: ', output_psfit_descriptors_dir]);
            end


 


            %获得sift的脸部的关键点
            image_face_getsift1 = imread(face_path); 
            gray_img = rgb2gray(image_face_getsift1);
            image_face_getsift = single(gray_img);
            [keypoints_noface, descriptors_noface] = vl_sift(image_face_getsift);


            % 获取所有关键点的 x 和 y 坐标
            x_noface = keypoints_noface(1,:);
            y_noface = keypoints_noface(2,:);

            % 检查坐标是否超出图像边界
            valid_indices = (x_noface >= 1) & (x_noface <= size(image_face_getsift, 2)) & (y_noface >= 1) & (y_noface<= size(image_face_getsift, 1));

            % 只保留在图像边界内的关键点
            keypoints_noface = keypoints_noface(:, valid_indices)';
            descriptors_noface = descriptors_noface(:, valid_indices)';


            % sift是无符号整数，在write_descriptors需要描述子是浮点型，现在改成浮点型
            descriptors_noface = double(descriptors_noface);

            %关键点的可视图
            featurePoints = keypoints_noface; 
            %featurePoints =   valid_keypoints'
            img = imread(face_path); 
            figure;
            imshow(img);
            hold on;     
            plot(keypoints_noface(:,1), keypoints_noface(:,2), 'g+');
            hold off;
            kps_photopath = sprintf('%d.png', cam);
            saveas(gcf, fullfile(output_dir, kps_photopath));
            filename = sprintf('%d.bmp.bin', cam);
            filename_pdescriptors = sprintf('%d.bmp.bin', cam);
            output_path = fullfile(output_dir, filename);
            output_path_pdescriptros = fullfile(output_psfit_descriptors_dir,filename_pdescriptors);
            write_keypoints(output_path,keypoints_noface)
            write_descriptors(output_path_pdescriptros,descriptors_noface)
            end
        end
    end
end


