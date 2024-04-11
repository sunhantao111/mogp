from read_data import read_arff
# from gp_nsga2 import gp_nsga2_classifier, count_selected_feat
import random
from functools import partial
from itertools import chain
from sklearn.metrics import roc_auc_score
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from baselinemethod import nsgp

from scipy.stats import kstest
from scipy import stats
from eval_func import aucc
from func_tools import *
import math
import time

resultspath = r'D:\file\pre-recall\result\biodata2_result'
dir_name = r'D:\file\pre-recall\uci\biodata2'
# methods = ['dist','ave','aucw','amse','G_mean']
# methods = ['dist','ave','amse']#corr,two_critrrion,aucw
# methods = ['corr','aucw','G_mean','dist','ave','amse']

# methods = ['auc_dist','corr','realauc','aucw','G_mean','dist','ave','amse']
methods = ['pre_recall']
# methods = ['G_mean','dist','ave','amse']
# all_fronts = []  # 存储每一次的front
# all_fitnesses = []  # 存储特征数和正确率
# all_fitnesses1 = []  # 存储几十次迭代过后的所有最优解
# tuple = ()
allpopauc = []
trainning_time_list = []
size_list = []


# all_f = []
# ach_list = []
# best_pop = []
# best_pop2 = []
# best_pop3 = []
# first_best_ensembles_auc = []
# second_best_ensembles_auc = []
# third_best_ensembles_auc = []
#file_container = ["ionosphere","wdbc","sonar","Leukemia","HillValley","Colon","liver-disorders",'diabetes']
#file_container = ['GSE14728','GSE14728','GSE30464','GSE42408', 'GSE46205','GSE76613', 'GSE145709']
# file_container = ["Colon","ionosphere","wdbc","sonar","Leukemia","HillValley","liver-disorders","diabetes","WBC","musk"]
#file_container = ["ionosphere","wdbc","sonar","Leukemia","HillValley","liver-disorders","diabetes"]
# file_container = ["Lymphoma","CNS","Colon","MLL","Ovarian","SRBCT"]
#file_container = ['liver-disorders','lapointe-2004-v2','colon','golub-1999-v1','leukemia','ionosphere','armstrong-2002-v1','shipp-2002-v1','dlbcl','laiho-2007','gordon-2002','yeoh-2002-v1',"Lymphoma",'su-2001','tomlins-2006','lung']
file_container = ['GSE14728','GSE30464','GSE42408', 'GSE46205','GSE65046','GSE71723','GSE76613', 'GSE98455','GSE145709']
# file_container =['GSE14728', 'GSE42408', 'GSE46205', 'GSE76613', 'GSE145709', 'GSE30464', 'GSE65046', 'GSE71723']


# bw = []
# for i in range(30):
#     a = random.randint(0, 32)
#     bw.append(a) 
bw = list(range(1, 31))
with open(resultspath + '.txt', 'a') as f:
    # f.write('pareto_ind_auc\t\tbest_auc\t\tach\t\t\tdistance\t\tensemble1\t\tensemble2\t\tensemble3\t\tdataset\n')
    f.write('method\t\t\t\tbest\t\t\t\tmean\t\t\t\tSTD\t\t\t\tfeatnum\t\t\t\tIR\t\t\t\tdataset\t\t\t\tselect_featnum\t\t\t\ttrainingtime\n')
    for file_name in file_container:
        for method in methods:

                                                                                                                                                                                                                                                                                    

            for circulation in range(30):
            

            
                # 读取数据
                # dir_name = "uci"
                # file_name = "wdbc"
                # 读取数据
                all_datas, flat_class = read_arff(dir_name, file_name)
                feat_num = len(all_datas[0]) - 1
                
                all_datas = np.array(all_datas)
                num_datas = all_datas[:, :-1]
                num_label = all_datas[:, -1]
                # print(all_datas[0])
                print(len(num_datas))
                print(len(num_label))

                # 划分数据集
                train_set,test_set,train_label,test_label = train_test_split(num_datas, num_label,
                                                                            train_size=0.7, test_size=0.3,
                                                                            random_state=bw[circulation], stratify=num_label)
                # 合并数据集和标签
                # 训练集
                all_datas1 = []
                for index, b in enumerate(train_label):
                    i = np.append(train_set[index], b)
                    i = i.tolist()
                    all_datas1.append(i)

                # 测试集
                all_datas2 = []
                for index, b in enumerate(test_label):
                    i = np.append(test_set[index], b)
                    i = i.tolist()
                    all_datas2.append(i)



                # 训练集
                majdatas1 = []
                mindatas1 = []
                for datas in all_datas1:
                    if datas[-1] == 1:
                        mindatas1.append(datas)
                    elif datas[-1] == 0:
                        majdatas1.append(datas)
                if len(majdatas1) <= len(mindatas1):
                    majdatas1, mindatas1 = mindatas1, majdatas1

                # 测试集
                majdatas2 = []
                mindatas2 = []
                for datas in all_datas2:
                    if datas[-1] == 1:
                        mindatas2.append(datas)
                    elif datas[-1] == 0:
                        majdatas2.append(datas)
                print(len(majdatas2), len(mindatas2))
                if len(majdatas2) <= len(mindatas2):
                    majdatas2, mindatas2 = mindatas2, majdatas2
                minnum = len(mindatas1)
                majnum = len(majdatas1)
                ir = majnum/minnum
                print("训练集少数类，多数类", minnum, majnum)
                test_num = (len(mindatas2), len(majdatas2))
                print("测试集少数类，多数类：", test_num)
                data_testing = mindatas2 + majdatas2
                N = (len(mindatas2),len(majdatas2))
                random.seed(circulation)
                start = time.time()
                pop,hof, toolbox,tools,pset = nsgp(method,all_datas1,majdatas1,mindatas1, feat_num,minnum,majnum)
                
                end = time.time()
                training_time = end -start
                trainning_time_list.append(training_time)
                fronts = tools.emo.sortNondominated(hof, len(hof))
                pareto_first_front = fronts[0]
                pareto_first_front = distinct(pareto_first_front)
                
                pareto_first_front = sorted(pareto_first_front,key = lambda ind:ind.fitness.values[0],reverse=True)
                best_ind = pareto_first_front[0]
                func = toolbox.compile(expr=best_ind)
                # pc_min = list(map(lambda a: func(a[:-1]), mindatas2))
                # pc_maj = list(map(lambda a: func(a[:-1]), majdatas2))
                size_list.append(count_selected_feat(best_ind,pset))
                auc = aucc(best_ind , toolbox, mindatas2, majdatas2, N)
                allpopauc.append(auc[0])

                # popfunc = list(map(lambda a: toolbox.compile(expr = a),pop))
                # change_data_testing = np.array(data_testing)
                # for ind in pop:

                #     auc = aucc(ind, toolbox, mindatas2, majdatas2, N)
                #     allpopauc.append(auc[0])
                # y = change_data_testing[:, -1]
                # for func in popfunc:
                #     myre = list(map(lambda a: func(a[:-1]), data_testing))
                #     myre = np.array(myre)
                #     myfit = roc_auc_score(y,myre)
                #     allpopauc.append(myfit)
                
                # print(len(allpopauc))
                # print(np.std(allpopauc,ddof = 1))
                print("第", circulation, "次循环")
            # 去重
        # 计算pareto front上所有分类器的auc均值
            aveauc = mean(allpopauc)
            bestauc = max(allpopauc)
            stdauc = np.std(allpopauc,ddof = 1)
            mean_training_time = mean(trainning_time_list)
            size = mean(size_list)
            print("30次平均auc是", aveauc)
            print("30次best_auc",bestauc)
            print('30次标准差为',stdauc)
            # st2 = p
        # print("40次平均best_auc3",b)

        # 写入结果
            f.write( str(method) +'\t\t\t\t'+ str(round(bestauc,4))
                + '\t\t\t\t'+str(round(aveauc,4))+'\t\t\t\t'+str(round(stdauc,4))
                 +'\t\t\t\t'+str(round(feat_num,4))+'\t\t\t\t'+str(round(ir,4))+'\t\t\t\t'+file_name+'\t\t\t\t'+str(size)+'\t\t\t\t'+str(mean_training_time)+'\n')
            f.flush()

        # 格式化所有存储结果的list
            del allpopauc[:]
            del trainning_time_list[:]
            del size_list[:]
            print('列表已清空：',allpopauc)
            # del ach_list[:]
            # del best_pop[:]
            # del best_pop2[:]
            # del best_pop3[:]
            # del first_best_ensembles_auc[:]
            # del second_best_ensembles_auc[:]
            # del third_best_ensembles_auc[:]
    f.close()
