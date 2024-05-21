import os
import random

folder_path = "/media/MD3400-2/Cailin/training_data/image"
all_files = os.listdir(folder_path)

train_percent = 0.8
num_files_80_percent = int(train_percent * len(all_files))
selected_files = random.sample(all_files, num_files_80_percent)
output_file_path = "/media/MD3400-2/Cailin/training_data/test.txt"

# 将文件名写入 txt 文件
with open(output_file_path, 'w') as output_file:
    for file_name in selected_files:
        output_file.write(file_name + '\n')

