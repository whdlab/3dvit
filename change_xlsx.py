import openpyxl

workbook = openpyxl.load_workbook("E:\\datasets\\data_forlenet\\AD_.xlsx")
worksheet = workbook.worksheets[0]

# # 在第一列之前插入一列
# worksheet.insert_cols(1)  #
a = worksheet['A2']
i = 1
for index, row in enumerate(worksheet.rows):
    if index == 0:
        continue  # 每一行的一个row[0]就是第一列


    if row[0].value == a:
        row[0].value += '_' + str(i)
        i += 1
    else:
        a = row[0].value
        i = 1
# 枚举出来是tuple类型，从0开始计数

workbook.save(filename="E:\\datasets\\data_forlenet\\AD_1.xlsx")



