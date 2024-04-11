# from functools import partial
# import numpy as np
# import matplotlib.pyplot as plt
# import pylab as pl
# from deap import creator
# from deap import base
# import numpy as np
#import datetime
# """plt.figure(1) # 创建图表1
# plt.figure(2) # 创建图表2
# ax1 = plt.subplot(221) # 在图表2中创建子图1
# ax2 = plt.subplot(222) # 在图表2中创建子图2
# x = np.linspace(0, 3, 100)
# for i in range(5):
#     plt.figure(1)  #❶ # 选择图表1
#     plt.plot(x, np.exp(i*x/3))
#     plt.sca(ax1)   #❷ # 选择图表2的子图1
#     plt.plot(x, np.sin(i*x))
#     plt.sca(ax2)  # 选择图表2的子图2
#     plt.plot(x, np.cos(i*x))
# plt.show()
# """
# '''
# X1 = range(0, 50)
# Y1 = [num**2 for num in X1] # y = x^2
# X2 = range(1,60,2)
# Y2 = [i*10for i in X2]  # y = x
# Fig = plt.figure(figsize=(8,4))                      # Create a `figure' instance
# Ax = Fig.add_subplot(111)               # Create a `axes' instance in the figure
# Ax.plot(X1, Y1,label = "zheng")                 # Create a Line2D instance in the axes
# Ax.plot(X2, Y2,label = "fu")
# Fig.legend()
# Fig.show()
# '''
# '''
# x = [1, 2, 3, 4, 5]# Make an array of x values
# y = [1, 4, 9, 16, 25]# Make an array of y values for each x value
# pl.plot(x, y,'b*')# use pylab to plot x and y
# pl.show()# show the plot on the scree
# '''
# '''x = [1, 2, 3, 4, 5]# Make an array of x values
# y = [1, 4, 9, 16, 25]# Make an array of y values for each x value
# pl.plot(x, y)# use pylab to plot x and y
# pl.title('Plot of y vs. x')# give plot a title
# pl.xlabel('x axis')# make axis labels
# pl.ylabel('y axis')
# pl.xlim(2.0, 4.0)# set axis limits
# pl.ylim(0.0, 30)
# pl.show()# show the plot on the screen
# '''
# # def subtraction(x, y):
# #     return x - y
# #
# #
# # f = partial(subtraction, 4)
# # print(f(5))
# # def multiply(x, y):
# #     return x * y
# #
# # def double(x, y=2):
# #     return multiply(x, y)
# # print(double(8))
# # m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# # n = [[1, 1, 1], [2, 2, 3], [3, 3, 3]]
# # print(list(zip(m,n)))
# # for a, b in zip(m, n):
# #     print(a,b)
# #creator.create("MultiObjMins", base.Fitness, weights=(-1.0, -1.0))
# from sklearn.model_selection import KFold
#
# import numpy as np
#
# X=np.array([[1,2],[3,4],[1,3],[3,5]])
#
# Y=np.array([1,2,3,4])
#
# KF=KFold(n_splits=4) #建立4折交叉验证方法 查一下KFold函数的参数
#
# for train_index,test_index in KF.split(X):
#
#     print("TRAIN:",train_index,"TEST:",test_index)
#
#     X_train,X_test=X[train_index],X[test_index]
#
#     Y_train,Y_test=Y[train_index],Y[test_index]
#
#     print(X_train,X_test)
#
#     print(Y_train,Y_test)
#
# # group_kfold = GroupKFold(n_splits=2)
# # for train_index, test_index in group_kfold.split(X, Y, groups):
# #     print("TRAIN:", train_index, "TEST:", test_index)
# #     X_train, X_test = X[train_index], X[test_index]
# #     Y_train, Y_test = Y[train_index], Y[test_index]
# #     print(X_train, X_test, Y_train, Y_test)
# '''y1=np.array(range(10))
# y2=np.array(range(20,30))
# m =np.append(y1,y2)
# y3 = np.array(np.random.randn(10))
# print(y3)
# m1 = np.append(m,y3)
# print(m1)
# n = [i//10 for i in range(30)]
# print(n)'''

# list2 = [value for value in range(3,31,3)]
# for a in list2:
#     print(a, end='\t')
# list1 = [value for value in range(1,1_000_0000)]
# starttime = datetime.datetime.now()
# print(sum(list1))
# endtime = datetime.datetime.now()
# print((endtime-starttime).seconds)

# def describe_pet(
#                 pet_name,animal_type = 'dog',eat="bone"
#                 ):
#     print(f"\nI have a {animal_type}.")
#     print(f"MY {animal_type}'s name is {pet_name.title()},and like eating {eat}.")
# describe_pet("willie")

# string = "ddddjjjjj" \
#         "dkkkdfsdfg"
# print(string)

# c = (3,88)
# b = (55,12)
# for a,d in zip(c,b):
#     print(a,"*****",d)
# basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
# print(basket)
# basket.add('apple')
# print(basket)
'''
# diget = ['1','2','3','?','5','86','3']
# diget1 = [item if item!='?' else '0'for item in diget ]
# print(diget1)
'''

# BI = [1,2,3,4,5]
# # CI = [3,4,5,6,4,6]
# # ZI = zip(BI,CI)
# #
# # # print("运用zip函数之后的",list(ZI))
#
# print("PI是",list(zip(*ZI)))
# A = "fjijijdsjoga"
# print(A.find("j"))
'''
from operator import itemgetter
import numpy
from deap import tools
len_stats = tools.Statistics(key=len)
itm0_stats = tools.Statistics(key=itemgetter(0))
mstats = tools.MultiStatistics(length=len_stats, item=itm0_stats)
mstats.register("mean", numpy.mean, axis=0)
mstats.register("max", numpy.max, axis=0)
print(mstats.items())
print(mstats.fields)
logbook = tools.Logbook()
k = mstats.fields
logbook.header = "gen",k[0],k[1]
print(logbook.header)
'''
'''
from operator import mul
a = [3,4,5]
b = [7,8]
c = tuple(map(mul,a,b))
print(c)
'''
'''
from deap import tools
hof = tools.ParetoFront()
print(list(hof))
print(list(hof) is None)
'''
# di = [0.0]*10
# print(di)
# def mysum(*args):
#     return sum(args)
# print(mysum([1,2],[1,4]))
'''
k = []
z = [1,2,3.3]

for a in range(1,6):
    k.append([a])
print(k)
print(len(k))

print(k.__repr__())
class person():
    def __init__(self,age,name):
        self.age = age
        self.name = name
    def __repr__(self):
        return "这是一个人类:名字叫做{0}，年龄：{1}".format(self.name,self.age)
per = person(25,"小王")
print(per)
'''

# sum = 0
# for i in range(1,6):
#     e = 0
#
#     if e == 2:
#         print("正确")
#     else:
#         sum += 1
#
#
#     print(sum)
#     continue





#
# from functools import partial
# def sum (a,b,c,f,d):
#     return a+b+c+d+f
# b= 5
# c= 6
# f= 6
# d= 9
# r = 8
# fun = partial(sum,b = b,c =c ,d = d, f= f )
# print(fun(7))
# fun2 = partial(sum,b = r)
# print(fun2(7))
# from deap import tools
# hof = tools.ParetoFront()
# print(len(hof))
# print(hof)
#
# a = [0,1,3]
# b = [0,1,5]
# a,b = b,a
# print(b)
# print("ddd",len(b))
#
#
#
# b.extend(a)
# print(b)
# all_f = []
# for i in range(1,6):
#     all_f.append(i)
# print(all_f)
# a = [(1,2),(5,2),(4,2),(3,2)]
# print(a[1][0])
#
# a.sort(key= lambda x:x[0])
# print(a)
#
# import math
#
# b = [1,2,2,3,6,5,5]
# c= max(b)
# print(c)
#
# a = math.sqrt(math.pow(5,2))
# print(a)

# a = open('F:/three objective/gp_nsga2/uci/WBC','r',encoding='utf-8')
# b = open("F:\\three objective\\gp_nsga2\\uci\\biodata\\WBC","w",encoding="utf-8")
#
# listd = a.readlines()
# listb = []
# for i in listd:
#     if i not in listb:
#         listb.append(i)
# b.writelines(listb)
#
# a.close()
# b.close()

# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# X, y = load_breast_cancer(return_X_y=True)
# print(X)
# clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
# b = clf.predict_proba(X)
#
# print("#"*50)
# print(b)
# a = roc_auc_score(y, clf.predict_proba(X)[:, 1])
# print(a)

# import random
# from numpy import mean
# a = [1,2,5,3,4]
# a.sort(reverse=True)
# print(a)
# print(mean(a))
# b = random.shuffle(a)
# print(type(b))
# import numpy as np
# from itertools import chain
#
#
#
# a = np.array([[1,1],[1,1],[1,1]])
# b = np.array([[2,2],[2,2],[2,2]])
# o = b.tolist()
# print(type(o))
# print(o)
# d = [[1,1],[1,1],[1,1]]
# e = [[2,2],[2,2],[2,2]]
#
# k = d+e
# c = a+b
# print(c)
# print(k)

# c = []
# print(a)
# print(b)
# for index,k in enumerate(b):
#     i = np.append(a[index],k)
#     c.append(i)
# print(c)
# print(a)

a = [(1,2),(1,3),(2,1),(2,2),(4,1),(3,2),(1,5)]
# b = max(a,key= lambda v:(v[1],v[0]))
# print(b)

for i,data in enumerate(a):
    pass
print(i)


