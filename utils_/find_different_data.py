# 读取文件1的数据
with open('C:\\Users\\whd\\Desktop\\annotations\\AD\\ad202.txt', 'r') as file1:
    data1 = file1.read().splitlines()

# 读取文件2的数据
with open('C:\\Users\\whd\\Desktop\\annotations\\AD\\ad248.txt', 'r') as file2:
    data2 = file2.read().splitlines()

# 查找在file2中存在但在file1中缺失的数据
missing_data = set(data2) - set(data1)
print(len(missing_data))
# 打印缺失的数据
for data in missing_data:
    print(data)
