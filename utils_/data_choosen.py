import csv

import numpy as np


def iterate_first_column(csv_file):
    list1 = []
    list2 = []
    with open(csv_file, 'r') as file:
        AD_tagetscsv = csv.reader(file)
        for row in AD_tagetscsv:
            cell_content = row[2]
            if cell_content != '':
                list1.append(cell_content)
            list2.append(row[0])
            # 在这里进行处理，可以输出、保存或执行其他操作
        print("current",len(list1))
        print("targets", len(list2))

    return list1, list2


Class_list = ["HC", "AD", "sMCI", "pMCI"]
for Class in Class_list:
    csv_file = 'C:\\Users\\whd\\Desktop\\annotations/{}_TOTAL.csv'.format(Class)  # 替换为实际的 CSV 文件路径

    a, b = iterate_first_column(csv_file)
    c = []
    d = []
    for i in b:
        if i[:10] not in a:
            c.append(i)
        else:
            d.append(i[:10])
    d_n = np.array(d)
    a_n = np.array(a)
    index_list = []
    for i in d_n:
        indexs = np.where(a_n==i)
        index_list.append(indexs[0][0])
        a.remove(i)
    aa =a_n[index_list]
    print("{}, {} both have:".format(Class, len(aa)), aa)
    print("{}, {} only in current:".format(Class, len(a)), a)

    with open('C:\\Users\\whd\\Desktop\\annotations\\{}_add{}.txt'.format(Class, len(c)), 'w') as file:
        file.writelines('\n'.join(c))
    with open('C:\\Users\\whd\\Desktop\\annotations\\{}_only_in_current{}.txt'.format(Class, len(a)), 'w') as file:
        file.writelines('\n'.join(a))
    print("Can add", len(c))

    print("TOTAL numbers of {} subject:".format(Class), len(a) + len(b) - len(aa))
