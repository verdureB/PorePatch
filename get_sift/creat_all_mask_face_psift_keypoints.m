clc
clear
close all

%vl库的路径所在
VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
file_content = fileread('/media/human_face_need_test/brown_v2/all_get_sift_parameters.txt');

% 按行分割
lines = strsplit(file_content, '\n');
%读入获取参数
% 初始化参数
py_path = '';
py_people = '';
no_kpt_py = 0;
select_frames = 2

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
            case 'select_frames'
                select_frames = str2num(value); % 转换为数值
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

            for frame=select_frames:select_frames
            %读入图片
            face_path = sprintf('%s/%s/%s/%d/psiftproject/images/%d.bmp', pre_path,people_num,emotion,frame,cam);
            image_face = imread(face_path); 
            
            %转换成灰度图便于后续的keypoints检测
            image_face = rgb2gray(image_face);      

            %读入mask
            mask_path = sprintf('%s/%s/%s/%d/psiftproject/mask/%d_mask.bmp', pre_path,people_num,emotion,frame,cam);
            image_mask = imread(mask_path);  
 
            %放缩因子，用于加速Psift关键点提取的速度
            scale=2;
            size(image_face);
            size(image_mask);
            size_p = size(image_face);

            %放大图片 和 对应的mask用于后续的检测关键点
            im_crop=func_maskPadding(image_face,image_mask,size_p(2)*scale,size_p(1)*scale); 

            % detect pore  检测毛孔关键点
            [frames,descriptors] = func_detect_pore_no_looser_boundaries(im_crop,no_kpt, 1);

            %创建文件夹用于存储
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

            %将前面检测到的关键点位置缩小回对应的原图比例，即在原本图像大小，关键点应该在的位置
            px=frames(1,:);
            py=frames(2,:);
            pscale=frames(3,:)/2;
            pori=frames(4,:);
            px = px / 2;
            py = py / 2;
            point = [px', py',pscale',pori'];
            %这里出来的descriptors原本就是double
            descriptors = descriptors';
            disp(size(descriptors))



            %关键点的可视图，并保存
            img = imread(face_path); 
            figure;
            imshow(img);
            hold on;     
            plot(point(:,1), point(:,2), 'g+');
            hold off;
            kps_photopath = sprintf('%d.png', cam);
            saveas(gcf, fullfile(output_dir, kps_photopath));
            %存储文件名
            filename = sprintf('%d.bmp.bin', cam);
            filename_pdescriptors = sprintf('%d.bmp.bin', cam);
            output_path = fullfile(output_dir, filename);
            output_path_pdescriptros = fullfile(output_psfit_descriptors_dir,filename_pdescriptors);
            %写成对应的句子
            write_keypoints(output_path,point)
            write_descriptors(output_path_pdescriptros,descriptors)
            end
        end
    end
end


