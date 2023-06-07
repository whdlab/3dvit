import os

import pandas as pd

# 读取 xlsx 文件
df = pd.read_excel('C:\\Users\\whd\\Desktop\\ad_diff.xlsx')
save_path = 'C:\\Users\\whd\\Desktop'
# 获取第一列和第三列的内容
column_list1 = df.iloc[:, 0].tolist()  # 第一列的内容转换为列表
column_list3 = df.iloc[:, 2].tolist()  # 第三列的内容转换为列表
# 将两个列表转换为集合
set1 = set(column_list1)
set3 = set(column_list3)

# 找出在第一列中独有的元素(str(x) for x in unique_elements_list3)
unique_elements_in_list1 = ','.join(str(x) for x in list(set1 - set3))

# 找出在第三列中独有的元素
unique_elements_in_list3 = ','.join(str(x) for x in list(set3 - set1))

# 保存独有的结果
with open(os.path.join(save_path, "only_in_one.txt"), 'w') as f:
    f.write(f"第一列独有的元素：{unique_elements_in_list1}\n"
            f"第三列独有的元素：, {unique_elements_in_list3}")
# 输出结果
print("第一列独有的元素：", unique_elements_in_list1)
print("第三列独有的元素：", unique_elements_in_list3)
