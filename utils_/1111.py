import re

data = """
  .\ADNI_022_S_0750_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070712155703378_S17695_I59552.nii,1
  .\ADNI_022_S_0750_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080321150601406_S46904_I99224.nii,1
  .\ADNI_023_S_0030_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070802171620543_S30858_I64059.nii,1
  .\ADNI_023_S_0030_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061204153813354_S8908_I31623.nii,1
  .\ADNI_023_S_0030_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081204144725253_S60282_I129251.nii,1
  .\ADNI_023_S_0042_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070802172507160_S33263_I64068.nii,1
  .\ADNI_023_S_0042_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070427182250489_S22275_I51939.nii,1
  .\ADNI_023_S_0042_MR_MT1__GradWarp__N3m_Br_20120402161713946_S135489_I294764.nii,1
  .\ADNI_023_S_0126_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061201141412569_S11525_I31271.nii,1
  .\ADNI_023_S_0126_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727132913188_S28717_I62422.nii,1
  .\ADNI_023_S_0126_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20100419173055100_S80628_I171407.nii,1
  .\ADNI_023_S_0126_MR_MT1__GradWarp__N3m_Br_20130503105553114_S185938_I370021.nii,1
  .\ADNI_023_S_0217_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070427195156610_S27364_I52075.nii,1
  .\ADNI_023_S_0217_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20090507112102895_S64059_I143348.nii,1
  .\ADNI_023_S_0217_MR_MT1__GradWarp__N3m_Br_20130313145653994_S151428_I363044.nii,1
  .\ADNI_023_S_0331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071226135653880_S19697_I86222.nii,1
  .\ADNI_023_S_0331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080207101213959_S41141_I89738.nii,1
  .\ADNI_023_S_0331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20100601182246872_S84476_I176275.nii,1
  .\ADNI_023_S_0331_MR_MT1__GradWarp__N3m_Br_20130503105912685_S186613_I370022.nii,1
  .\ADNI_023_S_0388_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061203115439863_S13076_I31437.nii,1
  .\ADNI_023_S_0388_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071113175153965_S42064_I81910.nii,1
  .\ADNI_023_S_0388_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20100601183225643_S82722_I176292.nii,1
  .\ADNI_023_S_0604_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061203122926229_S15182_I31455.nii,1
  .\ADNI_023_S_0604_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081026161028582_S54468_I123861.nii,1
  .\ADNI_023_S_0625_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080408125944312_S27184_I101445.nii,1
  .\ADNI_023_S_0625_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061203133805576_S16766_I31500.nii,1
  .\ADNI_023_S_0625_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20110222101321470_S75404_I218559.nii,1
  .\ADNI_023_S_0855_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061203140559204_S19011_I31518.nii,1
  .\ADNI_023_S_0855_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071006114755886_S38875_I77026.nii,1
  .\ADNI_023_S_0887_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061203141838289_S19087_I31526.nii,1
  .\ADNI_023_S_0887_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080414090014611_S47587_I102365.nii,1
  .\ADNI_023_S_0887_MR_MT1__GradWarp__N3m_Br_20120402162949430_S124406_I294803.nii,1
  .\ADNI_023_S_1126_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070828111153037_S23968_I70687.nii,1
  .\ADNI_023_S_1126_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080111141407076_S43053_I87086.nii,1
  .\ADNI_023_S_1247_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070412172006512_S25741_I48857.nii,1
  .\ADNI_023_S_1247_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080223124249591_S45054_I91703.nii,1
  .\ADNI_027_S_0179_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070807080551065_S28889_I65293.nii,1
  .\ADNI_027_S_0179_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071027172607283_S39424_I78876.nii,1
  .\ADNI_027_S_0835_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20100106110551519_S73784_I162381.nii,1
  .\ADNI_027_S_0835_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071027173307356_S39476_I78885.nii,1
  .\ADNI_027_S_0835_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081023104609073_S56475_I122963.nii,1
  .\ADNI_027_S_0835_MR_MT1__GradWarp__N3m_Br_20111229173957086_S123492_I274702.nii,1
  .\ADNI_027_S_1213_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080229160238156_S44650_I94453.nii,1
  .\ADNI_027_S_1213_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081017094717621_S54405_I121776.nii,1
  .\ADNI_027_S_1387_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070402182606620_S28276_I47757.nii,1
  .\ADNI_027_S_1387_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080408153947927_S47010_I101566.nii,1
  .\ADNI_027_S_1387_MR_MT1__GradWarp__N3m_Br_20120402165738848_S100917_I294861.nii,1
  .\ADNI_029_S_0878_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070805142357853_S18986_I64876.nii,1
  .\ADNI_029_S_0878_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080513174357765_S49230_I105342.nii,1
  .\ADNI_031_S_0294_MR_MPR-R__GradWarp__N3__Scaled_Br_20071226134606100_S12243_I86206.nii,1
  .\ADNI_031_S_0294_MR_MPR__GradWarp__N3__Scaled_Br_20071029190100730_S40106_I79604.nii,1
  .\ADNI_031_S_0568_MR_MPR-R__GradWarp__N3__Scaled_Br_20070904200910090_S33359_I71512.nii,1
  .\ADNI_031_S_0568_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080113121128788_S44131_I87247.nii,1
  .\ADNI_031_S_0568_MR_MPR__GradWarp__N3__Scaled_Br_20080626140237312_S51241_I111217.nii,1
  .\ADNI_031_S_1066_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080626142750653_S51293_I111257.nii,1
  .\ADNI_031_S_1066_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080220103435137_S23507_I90916.nii,1
  .\ADNI_031_S_1066_MR_MPR____N3__Scaled_Br_20090203114029042_S61032_I135106.nii,1
  .\ADNI_032_S_0187_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20071125130923129_S38352_I83061.nii,1
  .\ADNI_032_S_0187_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071030151315265_S20174_I79686.nii,1
  .\ADNI_032_S_0187_MR_MPR____N3__Scaled_Br_20070117234511672_S12179_I36440.nii,1
  .\ADNI_032_S_0214_MR_MPR____N3__Scaled_Br_20070117235730449_S19434_I36456.nii,1
  .\ADNI_032_S_0214_MR_MPR____N3__Scaled_Br_20080515115637792_S49571_I105747.nii,1
  .\ADNI_032_S_0214_MR_MT1__GradWarp__N3m_Br_20120402170955061_S105048_I294873.nii,1
  .\ADNI_032_S_0214_MR_MT1__GradWarp__N3m_Br_20160816122110975_S269901_I766944.nii,1
  .\ADNI_032_S_0978_MR_MPR____N3__Scaled_Br_20080116090946819_S44166_I87576.nii,1
  .\ADNI_033_S_0567_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070303120457162_S14572_I42370.nii,1
  .\ADNI_033_S_0567_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070813155156837_S33310_I67448.nii,1
  .\ADNI_033_S_0725_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070303130703791_S17092_I42409.nii,1
  .\ADNI_033_S_0725_MR_MPR__GradWarp__N3__Scaled_Br_20070303132129325_S17941_I42418.nii,1
  .\ADNI_033_S_0922_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070304122855389_S19341_I42493.nii,1
  .\ADNI_033_S_0922_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080501164738412_S48609_I104512.nii,1
  .\ADNI_033_S_0922_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20110221122807766_S93416_I218363.nii,1
  .\ADNI_033_S_0922_MR_MT1__GradWarp__N3m_Br_20131206154056450_S170588_I400483.nii,1
  .\ADNI_033_S_1116_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070813165712773_S33092_I67500.nii,1
  .\ADNI_036_S_0869_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070120011255669_S21421_I36993.nii,1
  .\ADNI_036_S_0869_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080613104743306_S49826_I109412.nii,1
  .\ADNI_036_S_0869_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20110222141716719_S94833_I219067.nii,1
  .\ADNI_037_S_0539_MR_MPR__GradWarp__N3__Scaled_Br_20070818133515805_S25820_I68538.nii,1
  .\ADNI_037_S_0539_MR_MPR__GradWarp__N3__Scaled_Br_20081023132948595_S52601_I123142.nii,1
  .\ADNI_037_S_0566_MR_MPR-R__GradWarp__N3__Scaled_Br_20081027081250795_S54743_I123891.nii,1
  .\ADNI_037_S_0566_MR_MPR__GradWarp__N3__Scaled_Br_20080224142834522_S45564_I92253.nii,1
  .\ADNI_037_S_0566_MR_MT1__GradWarp__N3m_Br_20130909122221376_S197306_I388454.nii,1
  .\ADNI_037_S_1225_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070921165941797_S37876_I74315.nii,1
  .\ADNI_041_S_0314_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061229141726504_S12492_I34680.nii,1
  .\ADNI_041_S_0314_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080108150608072_S43179_I86810.nii,1
  .\ADNI_041_S_0314_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20101109120045434_S85269_I203733.nii,1
  .\ADNI_041_S_1423_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20081017080327353_S55787_I121666.nii,1
  .\ADNI_051_S_1331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080530175403253_S49787_I107881.nii,1
  .\ADNI_051_S_1331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20100527161945092_S69898_I174851.nii,1
  .\ADNI_052_S_0952_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080207174038836_S20364_I89952.nii,1
  .\ADNI_052_S_0952_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071219112116310_S42607_I85533.nii,1
  .\ADNI_052_S_0952_MR_MT1__GradWarp__N3m_Br_20120409164230497_S96741_I296536.nii,1
  .\ADNI_052_S_1054_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070727103957599_S22955_I62234.nii,1
  .\ADNI_052_S_1054_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081020081018426_S52214_I122123.nii,1
  .\ADNI_053_S_0507_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080124152257197_S23928_I88430.nii,1
  .\ADNI_053_S_0507_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080703171303900_S51485_I112547.nii,1
  .\ADNI_057_S_0839_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070209151105220_S19188_I38670.nii,1
  .\ADNI_057_S_0839_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080421174018121_S48538_I103396.nii,1
  .\ADNI_067_S_0045_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061229171255332_S16966_I34783.nii,1
  .\ADNI_067_S_0077_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070816145120918_S27599_I68125.nii,1
  .\ADNI_067_S_0098_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070111013448257_S11324_I35920.nii,1
  .\ADNI_067_S_0098_MR_MPR__GradWarp__N3__Scaled_Br_20090714101933203_S65124_I148873.nii,1
  .\ADNI_067_S_0336_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20071221161459868_S32534_I85990.nii,1
  .\ADNI_067_S_0336_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070802203132429_S23837_I64314.nii,1
  .\ADNI_068_S_0442_MR_MPR____N3__Scaled_Br_20070821183920143_S28378_I69626.nii,1
  .\ADNI_068_S_0476_MR_MPR____N3__Scaled_Br_20070120021040408_S16883_I37034.nii,1
  .\ADNI_073_S_0518_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20081211113144637_S60430_I130072.nii,1
  .\ADNI_073_S_0518_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071221161742051_S43188_I85995.nii,1
  .\ADNI_082_S_0832_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20081105093003341_S57971_I125220.nii,1
  .\ADNI_082_S_0832_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080602083506500_S49031_I107907.nii,1
  .\ADNI_094_S_0434_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070210221747546_S22339_I39190.nii,1
  .\ADNI_094_S_0434_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080529132739527_S50352_I107717.nii,1
  .\ADNI_094_S_1015_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070818134949361_S32639_I68554.nii,1
  .\ADNI_098_S_0269_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070812145142247_S29778_I67232.nii,1
  .\ADNI_098_S_0269_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20090714103914383_S65766_I148880.nii,1
  .\ADNI_098_S_0667_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070818140815913_S27462_I68572.nii,1
  .\ADNI_098_S_0667_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081023161135149_S54853_I123262.nii,1
  .\ADNI_098_S_0667_MR_MT1__GradWarp__N3m_Br_20120411161950964_S126801_I297013.nii,1
  .\ADNI_099_S_0111_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070110234851842_S10933_I35849.nii,1
  .\ADNI_100_S_0892_MR_MPR____N3__Scaled_Br_20071103131157370_S30505_I80667.nii,1
  .\ADNI_100_S_0930_MR_MPR____N3__Scaled_Br_20070911121409167_S30198_I72526.nii,1
  .\ADNI_116_S_0649_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20100429103504102_S72494_I172347.nii,1
  .\ADNI_116_S_0649_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080229183619465_S45897_I94631.nii,1
"""

pattern = r'S\d+'
matches = re.findall(pattern, data)
field_list = [re.findall(pattern, line)[0] for line in data.split("\n") if re.findall(pattern, line)]

print(field_list)

import os

import os
import shutil


def find_and_copy_folders(source_folder, target_folder, folder_names):
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_folder):
        for folder in dirs:
            if folder in folder_names:
                folder_path = os.path.join(root, folder)
                target_path = os.path.join(target_folder, folder)
                shutil.copytree(folder_path, target_path)


# 设置源文件夹路径、目标文件夹路径和要查找的文件夹名称列表
source_folder = 'H:\\newpmci'
target_folder = 'D:\\dataset_fromE\\pMCI\\80loss'

# 调用函数进行查找和复制
find_and_copy_folders(source_folder, target_folder, field_list)
