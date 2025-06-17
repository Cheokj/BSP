import os
import sys

# input_file = '/public1/wangjing/Data_Seg/02_KiTS19/data/kits_train.txt'  # 输入文件路径
# output_file = '/public1/wangjing/Data_Seg/02_KiTS19/data/kits_pseudo-train.txt'  # 输出文件路径
# base_dir = '/public1/wangjing/ContinualLearning/output/02-kits/test_epoch_100/predict'  # 基础目录

input_file = sys.argv[1]
output_file = sys.argv[2]
base_dir = sys.argv[3]

with open(input_file, 'r') as f:
    lines = f.readlines()

with open(output_file, 'w') as f:
    for line in lines:
        line = line.strip()
        if line:  # 确保不是空行
            parts = line.split()
            image_path = parts[0]
            label_path = parts[1]

            # 从image_path中提取imaging-00000部分
            base_name = os.path.basename(image_path).replace('.nii.gz', '')

            # 生成新的第三个数据
            third_data = os.path.join(base_dir, base_name, base_name + '_pred.nii.gz')

            # 生成新的行
            new_line = f"{image_path} {label_path} {third_data}\n"
            f.write(new_line)

print(f"Processed lines written to {output_file}")