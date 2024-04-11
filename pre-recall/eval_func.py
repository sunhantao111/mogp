import numpy as np
import math
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import rankdata
from deap import gp
from func_tools import count_leaf_nodes,count_selected_feat
def Izt(r, k, c):
    """   
    :return: r,0
    """
    if k >= 0 > c:
        return r
    else:
        return 0
def dist(ind, toolbox, Cmin, Cmaj):
    """
    :return: 正负类平均数之间的距离，越大越好
    """
    # print(ind)
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        # print(len(Pc_min), type(Pc_min), Pc_min)
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        # print(len(Pc_maj), type(Pc_maj), Pc_maj)
        umin = np.mean(Pc_min)
        umaj = np.mean(Pc_maj)
        omin = np.std(Pc_min)
        omaj = np.std(Pc_maj)
        # print("umin:", umin)
        # print("umaj:", umaj)
        # print("omin:", omin)
        # print("omaj:", omaj)
        if omin+omaj == 0:
            return 0,
        else:
            result = (abs(umin-umaj)/(omin+omaj))*Izt(2, umin, umaj)
        # print("适应度函数值：", result)
        return result,
def operate_count(a, number, operator):
    num = 0
    if operator == '>=':
        for i in a:
            if i >= number:
                num += 1
    elif operator == '<':
        for i in a:
            if i < number:
                num += 1
    else:
        print('something wrong!')
    return num
def ave(ind, w, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    tpr = operate_count(Pc_min, 0, ">=")/min_num
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    tnr = operate_count(Pc_maj, 0, "<")/maj_num
    ave = w*tpr + (1-w)*tnr
    return ave,
def sig(x):
    # 对sigmoid函数的优化，避免了出现极大的数据溢出
    if x >= 0:
        return 2.0/(1+np.exp(-x))-1
    else:
        return (2*np.exp(x))/(1+np.exp(x))-1
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
    try:
        func([])
        return 0,
    except:
        Nmin = len(Cmin)
        Nmaj = len(Cmaj)
        k = [(0.5, Nmin, Cmin), (-0.5, Nmaj, Cmaj)]
        result = []
        for c in k:
            b = list(map(lambda a: pow(sig(func(a[:-1]))-c[0], 2)/(c[1]*2), c[2]))
            result.append(1-sum(b))
        result = sum(result)/2
        # print("适应度值为：", result)
        return result,
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
        myauc = sum(result)
    # return sum(result)
    return myauc,
#双标准函数
def two_criterion(individual,toolbox,majdatas,mindatas):
    func = toolbox.compile(expr=individual)
    majnum = len(majdatas)
    minnum = len(mindatas)
    mindata = []
    for i in mindatas:
        if i not in mindata:
            mindata.append(i)
    minnum1 = len(mindata)
    #c1
    iw = 0
    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    t = max(Pc_maj)
    for i in Pc_min:
        if i>t and i>=0:
            iw += 1
        else:
            pass
    c1 = iw/minnum1
    #c2
    umin = np.mean(Pc_min)
    umaj = np.mean(Pc_maj)
    u = (minnum*umin+majnum*umaj)/(minnum+majnum)
    pumin = list(map(lambda a:(a-u)**2),Pc_min)
    pumaj = list(map(lambda a:(a-u)**2),Pc_maj)
    pu = sum(pumin)+sum(pumaj)

    if pu == 0:
        c2 =0
    else:
        c2 = math.sqrt((minnum*(umin-u)**2+majnum*(umaj-u)**2)/pu)
    a_s = (c1,c2)
    return a_s
def Izrcorr(r,umin,umaj):
    if umin>=0 and umaj <0:
        return r
    else:
        return 0


def corr(individual,toolbox,majdatas,mindatas):

    func = toolbox.compile(expr=individual)
    majnum = len(majdatas)
    minnum = len(mindatas)

    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    umin = np.mean(Pc_min)
    umaj = np.mean(Pc_maj)
    u = (minnum*umin+majnum*umaj)/(minnum+majnum)
    pumin = list(map(lambda a:(a-u)**2,Pc_min))
    pumaj = list(map(lambda a:(a-u)**2,Pc_maj))
    pu = sum(pumin)+sum(pumaj)
    if pu == 0:
        r = 0
    else:
        r = math.sqrt((minnum*(umin-u)**2+majnum*(umaj-u)**2)/pu)
    corr = (r+Izrcorr(1, umin, umaj))/2
    return corr,
    

    pass
def G_mean(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    tpr = operate_count(Pc_min, 0, ">=")/min_num
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    tnr = operate_count(Pc_maj, 0, "<")/maj_num
    g_mean= math.sqrt(tpr + tnr)
    return g_mean,


            


    
def aucw(ind,toolbox,majdatas,mindatas):
    func = toolbox.compile(expr=ind)
    majdata = []
    mindata = []
    for i in majdatas:
        if i not in majdata:
            majdata.append(i)
    for i in mindatas:
        if i not in mindata:
            mindata.append(i)
        
    majnum = len(majdata)
    minnum = len(mindata)
    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    iw = 0

    multidata = float(majnum*minnum)
    for i in Pc_min:
        for j in Pc_maj:
            if i>j and i>=0:
                iw+=1
            else:
                pass
    iw = float(iw)
    aucww = iw/multidata
    return aucww,
def realauc(ind, toolbox, data_training):
    func = toolbox.compile(expr=ind)
    change_data_training = np.array(data_training)
    y = change_data_training[:, -1]
    myre = list(map(lambda a: func(a[:-1]), data_training))
    myre = np.array(myre)
    realauc = roc_auc_score(y,myre)
    return realauc,
def auc_dist(ind,toolbox,majdatas,mindatas):
    aucww = aucw(ind,toolbox,majdatas,mindatas)
    dist1 = dist(ind, toolbox,mindatas, majdatas)
    comp = count_leaf_nodes(ind)
    return aucww[0],dist1[0],comp

def prauc(ind, toolbox, data_training):
    func = toolbox.compile(expr=ind)
    change_data_training = np.array(data_training)
    y = change_data_training[:, -1]
    myre = list(map(lambda a: func(a[:-1]), data_training))
    myre = np.array(myre)
    prauc = average_precision_score(y,myre)
    return prauc,
def aucc(ind, toolbox, Cmin, Cmaj, N):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    o = np.array(Pc_min + Pc_maj)
    r = rankdata(o)
    sum_r_min = sum(r[:N[0]])
    auc = (sum_r_min - N[0]*(N[0]+1)/2)/(N[0]*N[1])
    return auc,
def count_terms(ind):
    num = 0
    for info in ind:
        if isinstance(info, gp.Terminal):
            num += 1
    return num
def nag(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 1, 0, 1000
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        fp = operate_count(Pc_maj, 0, ">=")
        tpr = tp / min_num
        fpr = fp / maj_num
        terms = count_terms(ind)
        return fpr, tpr, terms


def wang(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 1, 0
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        fp = operate_count(Pc_maj, 0, ">=")
        tpr = tp/min_num
        fpr = fp/maj_num
        return fpr, tpr

def pre_recall(individual,toolbox,pset,majdatas,mindatas):
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

def classes(func, datas):
    if func(datas[:-1]) >= 0:
        return 1.0
    else:
        return 0.0