import random
import operator

import itertools

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp



import matplotlib.pyplot as plt
import time

from func_tools import *





#  gp_nsga2多目标分类器
def gp_nsga2_classifier(majdatas1,mindatas1, feat_num, file_name, label,flat_class):

    # for i in range(len(data)):
    #     print("%s" % i, str(data[i]).rjust(20550, "-"))

    # defined a new primitive set for strongly typed GP
    #  创建一个迭代器，它返回指定次数的对象。如果未指定，则无限返回对象。
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, feat_num), float, "f")

    def Div(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(Div, [float, float], float)

    creator.create("MultiObjMin", base.Fitness, weights=(1.0, 1.0,-1.0)) #？？两个负一继承base.Fitness类
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.MultiObjMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # tree = toolbox.individual()
    # print("tree:", tree)
    # tree_str = str(tree)
    # print("str(tree):")
    # # print("func:", toolbox.compile(expr=tree))

    def classes(func, datas):
        if func(datas[:-1]) >= 0:
            return 1.0
        else:
            return 0.0

    # def classes(individual, datas):
    #     func = toolbox.compile(expr = individual)
    #     if func(datas[:-1]) >= 0:
    #         return 1.0
    #     else:
    #         return 0.0

   # 之前acc为适应度函数

   #  def evalfunc(individual, all_datas):
   #      # Transform the tree expression in a callable function
   #      func = toolbox.compile(expr=individual)
   #      # Randomly sample 400 mails in the spam database
   #      # random.sample()多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
   #      list = []
   #      for datas in all_datas:
   #          result = True if classes(func, datas) == datas[-1] else False
   #          list.append(result)
   #      acc = sum(list) / len(list)
   #      # comp = count_leaf_nodes(individual)  # 叶子节点数作为复杂度
   #      # comp = len(individual)   # 总的节点数作为复杂度
   #      comp = len(count_selected_feat(individual, pset))   # 选择出来的特征作为复杂度
   #
   #      # 约束
   #      # if comp > 1 and acc != 1:
   #      #     return -acc, comp
   #      # else:
   #      #     comp = 1000
   #      #     acc = 0
   #      #     return -acc, comp
   #
   #      return -acc, comp




#   PFC公式
    def Err(individual, class_datas):
        sum = 0
        for datas in class_datas:
            if classes(individual, datas) != datas[-1]:
                sum += 1
            else:
                pass

        return sum

    def compare_two_individual(ind1, ind2, datas):
        ind1fit = classes(ind1, datas)
        ind2fit = classes(ind2, datas)
        return 1 if ind1fit != ind2fit else 0

    def sum_compare_two_individual(ind1, ind2, class_datas):
        sum_difference = 0
        for datas in class_datas:
            sum_difference += compare_two_individual(ind1, ind2, datas)
        return sum_difference

    def sum_all_individual(poplution, individual, class_datas):
        sum_all = 0
        for ind in poplution:
            if ind is not individual:
                sum_all += (sum_compare_two_individual(ind, individual, class_datas)/
                           (Err(ind, class_datas)+Err(individual, class_datas)+1))

        return sum_all

    def pfc(population, npop, individual, class_datas):
        pfcvalue = (1/(npop-1)) * sum_all_individual(population,individual,class_datas)
        return pfcvalue

    def sig(x):
        # 对sigmoid函数的优化，避免了出现极大的数据溢出
        if x >= 0:
            return 2.0 / (1 + np.exp(-x)) - 1
        else:
            return (2 * np.exp(x)) / (1 + np.exp(x)) - 1

    def amse(ind, toolbox, Cmin, Cmaj):
        """
        :param ind:
        :param toolbox:
        :param k:
        :param Cmin:
        :param Cmaj:
        :return:
        """
        func = toolbox.compile(expr=ind)
        Nmin = len(Cmin)
        Nmaj = len(Cmaj)
        k = [(0.5, Nmin, Cmin), (-0.5, Nmaj, Cmaj)]
        result = []
        for c in k:
            b = list(map(lambda a: pow(sig(func(a[:-1])) - c[0], 2) / (c[1] * 2), c[2]))
            result.append(1 - sum(b))
        #result = sum(result) / 2
        # print("适应度值为：", result)
        return result


    # def evalfunc(individual,pop,npop,toolbox, all_datas):
    #     Cmin = [cmin for cmin in all_datas if cmin[-1] == 1.0]
    #     Cmaj = [cmaj for cmaj in all_datas if cmaj[-1] == 0.0]
    #     all = []
    #     all.append(Cmin)
    #     all.append(Cmaj)
    #     amselist = amse(individual, toolbox, Cmin, Cmaj)
    #     pfclist = []
    #     for c in all:
    #         pfcvalue = pfc(pop,npop,individual,c)
    #         pfclist.append(pfcvalue)
    #     w = 0.5
    #     cminvalue =list( map(lambda a, b: w * a + (1-w) * b,amselist,pfclist))
    #     return -cminvalue[0], -cminvalue[1]


    # def evalfunc(individual,majdatas,mindatas):
    #     ind = toolbox.compile(expr=individual)
    #     # majdatas = []
    #     # mindatas = []
    #     # for datas in all_datas:
    #     #     if datas[-1] == 0:
    #     #         mindatas.append(datas)
    #     #     elif datas[-1] == 1:
    #     #         majdatas.append(datas)
    #     # if len(majdatas) <= len(mindatas):
    #     #     majdatas, mindatas = mindatas, majdatas

    #     list1 = []
    #     list2 = []

    #     for datas in majdatas:
    #         result = True if classes(ind, datas) == datas[-1] else False
    #         list1.append(result)
    #     fpr = 1 - sum(list1)/len(list1)

    #     for datas in mindatas:
    #         result = True if classes(ind, datas) == datas[-1] else False
    #         list2.append(result)
    #     tpr = sum(list2)/len(list2)


    #     #comp = len(count_selected_feat(individual, pset))

    #     return round(fpr,6), round(tpr, 6)
    def evalfunc(individual,majdatas,mindatas):
        ind = toolbox.compile(expr=individual)
        # majdatas = []
        # mindatas = []
        # for datas in all_datas:
        #     if datas[-1] == 0:
        #         mindatas.append(datas)
        #     elif datas[-1] == 1:
        #         majdatas.append(datas)
        # if len(majdatas) <= len(mindatas):
        #     majdatas, mindatas = mindatas, majdatas

        list1 = []
        list2 = []

        for datas in majdatas:
            result = True if classes(ind, datas) == datas[-1] else False
            list1.append(result)
        fpr = 1 - sum(list1)/len(list1)

        for datas in mindatas:
            result = True if classes(ind, datas) == datas[-1] else False
            list2.append(result)
        tpr = sum(list2)/len(list2)


        #comp = len(count_selected_feat(individual, pset))

        return round(fpr,6), round(tpr, 6)
    

    def pre_recall(individual,majdatas,mindatas):
        ind = toolbox.compile(expr=individual)


        list1 = []
        list2 = []

        for datas in majdatas:
            result = True if classes(ind, datas) == datas[-1] else False
            list1.append(result)

        for datas in mindatas:
            result = True if classes(ind, datas) == datas[-1] else False
            list2.append(result)
        recall = sum(list2)/len(list2)
        try:
            precision = sum(list2)/(sum(list2)+len(list1)-sum(list1))
        except:

            precision = 0
        # comp = len(individual)


        comp = len(count_selected_feat(individual, pset))

        return round(recall,6), round(precision, 6),comp


    def tpr_fpr(threshold, list, Lmin, min_num, maj_num):
        tp = 0
        fp = 0
        for item in list:
            # print(item[0], threshold, item[1], Lmin)
            if item[0] >= threshold:
                if item[1] == Lmin:
                    tp += 1
                else:
                    fp += 1
        # print("-"*50, tp, fp)
        tpr, fpr = tp / min_num, fp / maj_num
        # print("-"*50, tpr, fpr)
        return tpr, fpr

    def my_auc(ind, toolbox, Cmin, Cmaj, test_num):
        datatesting = Cmin + Cmaj
        func = toolbox.compile(ind)
        outp = list(map(lambda a: func(a[:-1]), datatesting))
        label = [data[-1] for data in datatesting]
        outp_label = list(zip(outp, label))
        #print(outp_label)
        b = sorted(outp_label, key=(lambda x: x[0]))
        #print(b)
        tprs = []
        fprs = []
        c = list(set(outp))
        # print("-"*50, len(c))
        c.sort(reverse=True)
        # print("-" * 50, len(c), c)
        for threshold in c:
            tpr, fpr = tpr_fpr(threshold, b, 1, test_num[0], test_num[1])
            tprs.append(tpr)
            fprs.append(fpr)
        # print(tprs)
        # print(fprs)
        # graphl(fprs, tprs)
        result = []
        for i in range(len(c) - 1):
            s = (tprs[i + 1] + tprs[i]) * (fprs[i + 1] - fprs[i]) / 2
            # print("梯形的面积为：", s)
            result.append(s)
        # print("结果为：", result)
        return sum(result)

    def aCH(individuals, majdatas, mindatas):
        ach = 0
        achlist = []
        for index, ind in enumerate(individuals):
            ind.fitness.values = evalfunc(ind, majdatas, mindatas)
            print("第%d个是：" % index, ind.fitness.values)
            achlist.append(ind.fitness.values)
        achlist.append((1, 1))
        achlist.append((0, 0))
        achlist.sort(key=lambda x: (x[0], x[1]))
        print(achlist)
        for i in range(len(achlist)-1):
            ach += (achlist[i + 1][1] + achlist[i][1]) * (achlist[i+1][0] - achlist[i][0]) / 2
        return ach

    toolbox.register("evaluate", pre_recall, majdatas=majdatas1, mindatas=mindatas1)
    #toolbox.register("PFC",pfc,all_datas = all_datas)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("selectGen1", tools.selTournament, tournsize=2)
    toolbox.register('select', tools.emo.selTournamentDCD)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 装饰器  使用指定的装饰器装饰别名，别名必须是当前工具箱中的已注册函数。
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    # random.seed(10)
    N_POP = 500
    N_GEN = 50
    CXPB = 0.9  # 交叉概率，参数过小，族群不能有效更新
    MUTPB = 0.1  # 突变概率，参数过小，容易陷入局部最优
    pop = toolbox.population(n=N_POP)

    # for i, ind in enumerate(pop):
    #     if ind.fitness.values:
    #         print("个体%d:" % (i+1), len(ind), ind.fitness.values, ind)
    #     else:
    #         print("个体%d:" % (i+1), len(ind), (0, 0, 0), ind)

    # 统计
    stats_recall = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_pre = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_comp = tools.Statistics(lambda ind: ind.fitness.values[2])
    # stats_comp = tools.Statistics(lambda ind: ind.fitness.values[2])

    mstats = tools.MultiStatistics(recall=stats_recall, precision=stats_pre,comp = stats_comp)

    print("多目标统计器的三个目标为：", mstats.keys())
    mstats.register("avg", np.average)
    # mstats.register("std", np.std)
    mstats.register("max", np.max)
    # mstats.register("median", np.median)
    mstats.register("min", np.min)
    hof = tools.ParetoFront()  # 名人堂
    #print(hof)

    #toolbox.register("evaluate", evalfunc, pop=pop, npop=N_POP, toolbox=toolbox, all_datas=all_datas)


    pop,hof = algorithms.eaNSGA2(pop, toolbox, CXPB, MUTPB, N_GEN, N_POP, stats=mstats, halloffame=hof, verbose=True)
    # f = distinct(hof)
    fronts = tools.emo.sortNondominated(pop, len(pop))
    # for i, front in enumerate(fronts):  # 使用枚举循环得到各层的标号与pareto解
    #     print("pareto非支配等级%d解的个数：" % (i+1), len(front), front)
    pareto_first_front = fronts[0]  # 返回的不同前沿的pareto层集合fronts中第一个front为当前最优解集
    pareto_first_front = distinct(pareto_first_front)#去除重复的个体
    #print("gggg:",len(pareto_first_front))

    #return pareto_first_front, evalfunc, tools, pset
    return pareto_first_front,evalfunc, my_auc,aCH, toolbox,tools,pset



