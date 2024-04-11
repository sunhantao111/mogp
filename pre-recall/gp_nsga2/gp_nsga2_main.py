from read_data import read_arff
from gp_nsga2 import gp_nsga2_classifier, count_selected_feat
import random
from functools import partial
from itertools import chain
from sklearn.metrics import roc_auc_score
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from func_tools import *

resultspath = 'D:\\fpr-tpr\\gp_nsga2'
dir_name = 'D:\\fpr-tpr\\gp_nsga2\\uci\\biodata2'

# all_fronts = []  # 存储每一次的front
# all_fitnesses = []  # 存储特征数和正确率
# all_fitnesses1 = []  # 存储几十次迭代过后的所有最优解
# tuple = ()
bw = list(range(1, 41))
feature_list = []
all_f = []
ach_list = []
best_pop = []
best_pop2 = []
best_pop3 = []
first_best_ensembles_auc = []
second_best_ensembles_auc = []
third_best_ensembles_auc = []
#file_container = ["ionosphere","wdbc","sonar","Leukemia","HillValley","Colon","liver-disorders"]
file_container = ['GSE14728','GSE14728','GSE30464','GSE42408', 'GSE46205','GSE76613', 'GSE145709']
#file_container = ["ionosphere","wdbc","sonar","Leukemia","HillValley"]
with open(resultspath + 'machine5.txt', 'a') as f:
    f.write('best_auc\t\t\t\tmean_auc\t\t\t\toriginal_features\t\t\t\tave_select_features\t\t\t\tdataset\n')
    for file_name in file_container:


        for circulation in range(40):
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

            # # 70%训练集 30%测试集
            # data_traing = random.sample(all_datas, int(len(all_datas)*0.7))
            # data_testing = [i for i in all_datas if i not in data_traing]

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

            # 没有固定训练集和测试集
            # mindatas = []
            # majdatas = []
            # for datas in all_datas:
            #     if datas[-1] == 1:
            #         mindatas.append(datas)
            #     elif datas[-1] == 0:
            #         majdatas.append(datas)
            # print(len(majdatas), len(mindatas))
            # if len(majdatas) <= len(mindatas):
            #     majdatas, mindatas = mindatas, majdatas

            # majdatas1 = random.sample(majdatas, int(len(majdatas) * 0.7))
            # majdatas2 = [i for i in majdatas if i not in majdatas1]
            # mindatas1 = random.sample(mindatas, int(len(mindatas) * 0.7))
            # mindatas2 = [i for i in mindatas if i not in mindatas1]

            print("训练集少数类，多数类", len(mindatas1), len(majdatas1))
            test_num = (len(mindatas2), len(majdatas2))
            print("测试集少数类，多数类：", test_num)
            data_testing = mindatas2 + majdatas2

            # kfold 交叉验证
            # p = 0
            # kf = KFold(n_splits=10,random_state=None)
            # for training_index,testing_index in kf.split(all_datas):
            # data_traing,data_testing = np.array(all_datas)[training_index],np.array(all_datas)[testing_index]

            # training
            label = "训练集"
            pareto_first_front, evalfunc,my_auc, aCH, toolbox, tools, pset = gp_nsga2_classifier(majdatas1, mindatas1,
                                                                                          feat_num, file_name, label,
                                                                                        flat_class)
            pareto_first_front = sorted(pareto_first_front, key=lambda ind:ind.fitness.values[0])
            train_first_paretofront = []

            for ind in pareto_first_front:
                train_first_paretofront.append(ind.fitness.values)
            # print(train_first_paretofront)

            # graph(train_first_paretofront, file_name)

            # evalfunc = partial(evalfunc,majdatas = majdatas2,mindatas = mindatas2)
            # test_result = list(map(evalfunc,pareto_first_front))
            # print(test_result)
            # print("~" * 30 + "测试前" + "~" * 30)
            # for i, ind in enumerate(pareto_first_front):
            #     print("第%d个" % i, ind, ind.fitness.values)
            # graph(test_result, file_name)

            # 寻找斜率范围内的最好集成分类器
            # min_sloap = -4
            # max_sloap = -1/4
            # best_ensembles = find_best_ensembles(pareto_first_front,min_sloap,max_sloap)
            # print("~" * 30)
            # print("the number of sloap_ensemble:",len(best_ensembles))
            # # print(best_ensembles)
            # for z in best_ensembles:
            #     print(z,z.fitness.values)

            # for i,ind in enumerate(pareto_first_front):
            #     print("第%d：" %i,ind,ind.fitness.values)
            # print("the number of pareto front：", len(pareto_first_front))

            # # 删除重复目标值个体，选择复杂度最少的集成
            # unrepeat_ensembles = find_unrepeat_ensembles(pareto_first_front,pset)
            # print("the number of unrepeat_ensembles:",len(unrepeat_ensembles))

            # 寻找最好的个体1
            # best_ind = find_best_individual(pareto_first_front,pset)
            # print(best_ind,best_ind.fitness.values)
            best_ind = find_individual(pareto_first_front,pset)
            select_features = len(count_selected_feat(best_ind, pset))
            feature_list.append(select_features)


            # 寻找最优个体方法2
            # k = len(majdatas1)/len(mindatas1)
            # best_ind2 = find_min_distance(1,1,pareto_first_front,pset)
            # print(best_ind2,best_ind2.fitness.values)


            # testing
            # label = "测试集"
            # print("-" * 30 + "测试" + "-" * 30)
            # # partial 函数的功能就是：把一个函数的某些参数给固定住,且在函数调用的时候不能修改，返回一个新的函数。
            # # evalfunc_partial = partial(evalfunc, majdatas=majdatas2, mindatas=mindatas2)
            # test_evalufunc = partial(my_auc, toolbox=toolbox, Cmin=mindatas2, Cmaj=majdatas2, test_num=test_num)
            # # evalfunc(ind, all_datas=data_testing)
            # fitnesses = map(test_evalufunc, pareto_first_front)
            # fitnesses = map(evalfunc_partial, pareto_first_front)
            # for i, fit in enumerate(fitnesses):
            #     print("个体%d的适应度为：" % i, fit)

            # 最好的个体AUC:方法一
            best_fit = my_auc(best_ind, toolbox, mindatas2, majdatas2, test_num)
            print("***" * 50)
            print("best_ind的auc值是：", best_fit)
            print("***" * 50)
            best_pop.append(best_fit)
            ave_select_features = mean(feature_list)

            # 最好的个体AUC：方法二
            # best_ind = toolbox.compile(best_ind)
            # # best_ind2 = toolbox.compile(best_ind2)
            # re = list(map(lambda a: best_ind(a[:-1]), data_testing))
            # # re2 = list(map(lambda a: best_ind2(a[:-1]), data_testing))
            # change_data_testing = np.array(data_testing)
            # y = change_data_testing[:, -1]
            # re = np.array(re)
            # # re2 = np.array(re2)
            # best_fit2 = roc_auc_score(y, re)
            # # best_fit3 = roc_auc_score(y, re2)
            # best_pop2.append(best_fit2)
            # print(best_fit2)
            # best_pop3.append(best_fit3)

            # 计算集成的auc：方法一
            # first_final_output = simple_average_ensemble(best_ensembles,data_testing,toolbox)
            # first_final_output = np.array(first_final_output)
            # k = roc_auc_score(y,first_final_output)
            # first_best_ensembles_auc.append(k)

            # # 计算集成的auc：方法二
            # second_final_output = vote_ensemble(best_ensembles,data_testing,toolbox)
            # second_final_output = np.array(second_final_output)
            # j = roc_auc_score(y,second_final_output)
            # second_best_ensembles_auc.append(j)

            # print("集成auc1：",k)
            # print("集成auc20",j)

            # # 计算集成的auc：方法三
            # third_final_output = simple_average_ensemble(unrepeat_ensembles, data_testing, toolbox)
            # third_final_output = np.array(third_final_output)
            # w = roc_auc_score(y, third_final_output)
            # third_best_ensembles_auc.append(w)

            # 统计pareto上所有分类器的auc值
            # for i, value in enumerate(fitnesses):
            #     # print("第%d个适应度值:" % i, value)
            #     all_f.append(value)

            #计算ach
            # ach = aCH(pareto_first_front, majdatas2, mindatas2)
            # print("ach:", ach)
            # ach_list.append(ach)

            # for ind, fitness in zip(pareto_first_front, fitnesses):
            #     ind.fitness.values = fitness
            #     print(ind, ind.fitness.values)

            # pareto_first_front = tools.sortNondominated(pareto_first_front, len(pareto_first_front), first_front_only=True)
            # # print("dddd%d",len(pareto_first_front))
            # pareto_first_front = pareto_first_front[0]
            # for i, ind in enumerate(pareto_first_front):
            #     print("个体%d及其适应度值为：" % (i+1), ind, ind.fitness.values)
            # for ind in pareto_first_front:
            #     all_fronts.append(ind)

            print("第", circulation, "次循环")
            # 去重
        # 计算pareto front上所有分类器的auc均值
        # ave_pareto_fitness = mean(all_f)
        # 计算40次ach的平均值
        # ave_ach = mean(ach_list)

        # 计算40次auc:离（0,1）点最近的个体
        # a = mean(best_pop2)

        # 计算40次auc:离等性能线最近的个体
        # b = mean(best_pop3)

        # my_auc,40次平均best_auc：离离（0,1）点最近的个体
        ave_best_pop = mean(best_pop)
        best_auc = max(best_pop)

        # print("40次平均auc是", ave_pareto_fitness)
        # print("40次平均ROCCH", ave_ach)
        print("my_auc,40次平均best_auc:", ave_best_pop)
        # print("40次平均best_auc2",a)
        # print("40次平均best_auc3",b)

        # 集成平均auc
        # ensembles = mean(first_best_ensembles_auc)
        # ensembles2 = mean(second_best_ensembles_auc)
        # ensembles3 = mean(third_best_ensembles_auc)
        print(best_auc)
        print(ave_best_pop)
        print(feat_num)
        print(ave_select_features)

        # 写入结果
        f.write(str(round(best_auc,4))+'\t\t\t\t\t'+str(round(ave_best_pop,4))
                +'\t\t\t\t\t\t'+str(feat_num)+'\t\t\t\t\t\t\t'+str((round(ave_select_features,4)))
                 +'\t\t\t\t\t\t\t'+file_name+'\n')
        f.flush()

        # 格式化所有存储结果的list
        del all_f[:]
        del ach_list[:]
        del best_pop[:]
        del best_pop2[:]
        del best_pop3[:]
        del first_best_ensembles_auc[:]
        del second_best_ensembles_auc[:]
        del third_best_ensembles_auc[:]


        # #全部最优解pareto排序之前
        # all_fronts = distinct(all_fronts)
        # for ind in all_fronts:
        #     temp = (ind.fitness.values[1],ind.fitness.values[0])
        #     all_fitnesses1.append(temp)
        # graph(all_fitnesses1, file_name)

        # 几十次后对全部最优模型进行非支配排序
        # muti_time_pareto_first_front = tools.sortNondominated(all_fronts, len(all_fronts), first_front_only=True)
        # muti_time_pareto_first_front = muti_time_pareto_first_front[0]

        # all_fronts_new = init_two_dimensional_list(100)
        # 按照特征数把个体存储在一个二维数组中
        # for ind in all_fronts:
        #     #print(ind, ind.fitness.values[1])
        #     index = int(ind.fitness.values[1]) - 1
        #     all_fronts_new[index].append(ind)

        # #30次平均acc和特征数
        # for i, front in enumerate(all_fronts_new):
        #     acc_sum = 0.0
        #     if front:
        #         for ind in front:
        #             acc_sum += ind.fitness.values[0]
        #         acc = acc_sum/len(front)
        #         tuple = ((i+1), acc)
        #         all_fitnesses.append(tuple)
        # print(all_fitnesses)
        #
        # sorted_all_fitnesses = my_nondominedsort(all_fitnesses,len(all_fitnesses),first_front_only=True)
        # fitness_first_paretofront = sorted_all_fitnesses[0]
        # print(fitness_first_paretofront)

        # 图形化显示
        # graph(fitness_first_paretofront, file_name)
    f.close()
