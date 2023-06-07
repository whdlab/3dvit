file_path = "C:\\Users\\whd\\Desktop\\annotations\\AD\\missing_mmse_values.txt"  # 替换为您的文本文件路径
list = []
with open(file_path, "r") as file:
    for i, line in enumerate(file):
        line = line.strip()  # 去除行尾的换行符和空格
        line = line[:]
        if i != 0:
            list.append(',' + line)
        else:
            list.append(line)
        print(line)  # 打印每一行文本


file_path = "C:\\Users\\whd\\Desktop\\annotations\\AD\\sort_missing_mmse_values.txt"  # 替换为您要写入的文本文件路径

print(len(list))
with open(file_path, "a") as file:
    for line in list:
        file.write(line)  # 写入每一行文本，并在末尾加上换行符
