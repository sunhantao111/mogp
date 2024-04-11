from scipy.io import arff
import pandas as pd 

file_name=r'D:\file\two_criterion\datasets/Colon.arff'

data,meta=arff.loadarff(file_name)
#print(data)
# print(meta)

df=pd.DataFrame(data)
# 样本
sample = df.values[:, 0:len(df.values[0])-1]
# 对标签进行处理
# [b'1' b'-1' ...]bytes类型
label = df.values[:, -1] # 要处理的标签
cla = [] # 处理后的标签
for i in label:
    test = int(i)
    cla.append(test)
print(cla)
# print(df.head())
# print(df.values)
#保存为csv文件
# out_file='/Users/schillerxu/Documents/sourcecode/python/pandas/CM1.csv'
# output=pd.DataFrame(df)
# output.to_csv(out_file,index=False)
