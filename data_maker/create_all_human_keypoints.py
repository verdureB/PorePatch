import os
import subprocess
from tqdm import tqdm
data_path = '/media/human_face_need_test/PsPD'
human_list = [
    item
    for item in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, item)) # 判断是否为目录
]
sorted_human_list = sorted(human_list, key=int)
select_frames = 2
detect_keypoints_number = 10000


# 用于读取记录人物制作文件的函数
import re
def extract_numbers_from_txt(filepath):
    """从txt文件中提取前面的编号。
    Args:
        filepath: txt文件的路径。
    Returns:
        一个包含提取出的编号的列表，如果文件不存在或发生错误，则返回空列表。
    """
    numbers = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = re.match(r'(\d+)', line)  # 使用正则表达式匹配数字
                if match:
                    numbers.append(match.group(1))
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return numbers

already_create_keypoints_txt_path = os.path.join(data_path,"Finished_create_keypoints.txt")

# 获取人物编号并排序
already_create_keypoints_human_list = extract_numbers_from_txt(already_create_keypoints_txt_path)

need_create_cloudply_human_list = [item for item in sorted_human_list if item not in already_create_keypoints_human_list]

for cur_human in tqdm(need_create_cloudply_human_list):
    print(f"当前执行人物:{cur_human}")
    # 人物编号数据集所在位置
    human_full_path =  os.path.join(data_path,cur_human,'EMO-1-shout+laugh',str(select_frames))
    project_path = os.path.join(human_full_path,"psiftproject")
    project_images_path= os.path.join(project_path,"images")
    project_mask_path= os.path.join(project_path,"mask")

    #进行keypoint检测代码，用的是psfit
    dataset_path_call_matlab = data_path
    get_sift_human_number =  cur_human
    kpts_number = detect_keypoints_number
    with open("all_get_sift_parameters.txt", "w") as f:
        f.write(f"dataset_path_call_matlab={dataset_path_call_matlab}\n")
        f.write(f"human_number={get_sift_human_number}\n")
        f.write(f"kpts_number={kpts_number}\n")
        f.write(f"kpts_number={select_frames}\n")

    #调用matlab
    get_sift_m='/media/dataset_maker_matlab_python/get_sift/creat_all_mask_face_psift_keypoints.m'
    log_file = os.path.join(project_path, "matlab_output_getkeypoints.log")

    with open(log_file, "w") as outfile:
        subprocess.run([
            "/usr/local/Matlab/R2020a/bin//matlab",
            "-nodisplay", "-nosplash",
            "-logfile", log_file,
            "-r", "run('{}'); exit".format(get_sift_m)
        ], stdout=outfile, stderr=outfile)  # 将 stdout 和 stderr 都重定向到日志文件
        
    with open(os.path.join(data_path,"Finished_create_keypoints.txt"), "a") as f:
        f.write(f"{cur_human} Finish create keypoints!\n")