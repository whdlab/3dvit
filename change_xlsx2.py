import openpyxl

workbook = openpyxl.load_workbook("E:\\datasets\\data_forlenet\\AD_1.xlsx")
worksheet = workbook.worksheets[0]

for index, row in enumerate(worksheet.rows):
    if index == 0:
        continue  # 每一行的一个row[0]就是第一列
    if len(row[0].value) == 10:
        row[0].value += '_0'

workbook.save(filename="E:\\datasets\\data_forlenet\\AD_11.xlsx")