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

import pygraphviz as pgv
from func_tools import *
from eval_func import *





#  gp_nsga2多目标分类器
def nsgp(method,all_datas1,majdatas1,mindatas1, feat_num,minnum,majnum):
    N = [minnum,majnum]

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


    creator.create("Fitnessmax", base.Fitness, weights=(1.0, 1.0,-1.0)) #？？两个负一继承base.Fitness类
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitnessmax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    if method == 'pre_recall':
        toolbox.register("evaluate", pre_recall,toolbox = toolbox, pset = pset,majdatas = majdatas1,mindatas = majdatas1)

    


    else:
        print('wrong method')




    # toolbox.register("evaluate", evalfunc, majdatas=majdatas1, mindatas=mindatas1)
    #toolbox.register("PFC",pfc,all_datas = all_datas)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("selectGen1", tools.selTournament, tournsize=3)
    toolbox.register("selectGen1", tools.selTournament, tournsize=7)
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
    CXPB = 0.8  # 交叉概率，参数过小，族群不能有效更新
    MUTPB = 0.2  # 突变概率，参数过小，容易陷入局部最优

    pop = toolbox.population(n=N_POP)
    stats_fpr = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_tpr = tools.Statistics(lambda ind: ind.fitness.values[1])
    #stats_comp = tools.Statistics(lambda ind: ind.fitness.values[2])
    # stats_comp = tools.Statistics(lambda ind: ind.fitness.values[2])

    mstats = tools.MultiStatistics(fpr=stats_fpr, tpr=stats_tpr)

    print("多目标统计器的两个目标为：", mstats.keys())

    # mstats = tools.MultiStatistics(c1=stats_c1, c2=stats_c2)

    # print("多目标统计器的两个目标为：", mstats.keys())
    # print("多目标统计器的两个目标为：", mstats.keys())
    # mstats.register("avg", np.average,axis=0)
    # # mstats.register("std", np.std)
    # mstats.register("max", np.max,axis=0)
    # # mstats.register("median", np.median)
    # mstats.register("min", np.min,axis=0)
    mstats.register("avg", np.average)
    # mstats.register("std", np.std)
    mstats.register("max", np.max)
    # mstats.register("median", np.median)
    mstats.register("min", np.min)
    # hof = tools.HallOfFame(1)  # 名人堂
    #print(hof)
    hof = tools.ParetoFront()

    #toolbox.register("evaluate", evalfunc, pop=pop, npop=N_POP, toolbox=toolbox, all_datas=all_datas)


    pop,hof= algorithms.eaNSGA2(pop, toolbox, CXPB, MUTPB, N_GEN, N_POP, stats=mstats, halloffame=hof, verbose=True)
    #print("名人堂：", len(hof), hof, type(hof))
    # f = distinct(hof)


    #print("gggg:",len(pareto_first_front))

    #return pareto_first_front, evalfunc, tools, pset
    return pop,hof, toolbox,tools,pset



