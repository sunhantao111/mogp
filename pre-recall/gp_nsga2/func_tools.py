import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import math


# 统计叶子结点个数
def count_leaf_nodes(ind):
    ind_str = str(ind)
    leaf_nodes = ind_str.count("f")
    return leaf_nodes


# 统计选择出来的特征
def count_selected_feat(ind, pset):
    list = []
    ind_str = str(ind)
    for f in reversed(pset.arguments):
        if ind_str.find(f) != -1:
            list.append(f)
            ind_str = ind_str.replace(f, '')
    return list


# 生成二维空列表
def init_two_dimensional_list(rows):
    list = []
    for row in range(rows):
        list.append([])
    return list


def graph(list, file_name):
    list1 = []
    list2 = []
    for comp ,acc in list:
        list1.append(comp)
        list2.append(acc)

    #print(list2)


    #plt.annotate('局部最大', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.grid(axis = 'y')
    for a ,b in list:
         plt.text(a,b,'%.4f' % b)
    plt.plot(list1, list2)
    # for comp, acc in list:
    #     plt.plot(acc, comp, "r--", marker="*", lw=2)
    #matplotlib将使用rcParams字典中的配置进行绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(file_name + '-pareto前沿')
    plt.xlabel('特征数')
    # plt.ylabel('复杂度')
    plt.ylabel('-准确率')
    plt.axis([0,1,0,1])
    #plt.tight_layout()
    plt.grid()
    plt.savefig(f"C:\\Users\89301\\Desktop\\data image3\\{file_name}")
    plt.show()


def graph_inviduals(inviduals, file_name, label):
    for i, ind in enumerate(inviduals):
        # print("种群中最优个体%d及其适应度值为：" % (i + 1), ind, ind.fitness.values)
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "r--", marker="o", lw=2)
        # plt.scatter(ind.fitness.values[0], ind.fitness.values[1], marker="o", s=10)
    # for ind in pareto_first_front:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(file_name + '-pareto前沿' + "(" + label + ")")
    plt.xlabel('-准确率')
    # plt.ylabel('复杂度')
    plt.ylabel('特征数')
    plt.tight_layout()
    plt.grid()
    plt.show()


# 去重
def distinct(pareto_first_front):
    pareto_first_front_str = map(str, pareto_first_front)
    pareto_first_front_str = set(pareto_first_front_str)
    # for ind in pareto_first_front_str:
    #     print(ind)
    pareto_no_repeat = []
    for ind in pareto_first_front:
        if str(ind) in pareto_first_front_str:
            pareto_no_repeat.append(ind)
            pareto_first_front_str.remove(str(ind))
    return pareto_no_repeat

#支配关系确认：特征数和正确率组成
def my_dominates (onepoint, otherpoint):

    not_equal = False
    for one_point, other_point in zip(onepoint, otherpoint):
        if one_point < other_point:
            not_equal = True
        elif one_point > other_point:
            return False
    return not_equal

#对于特征数和正确率非支配排序
def my_nondominedsort(individuals, k,first_front_only):
    fits = []
    for ind in individuals:
        fits.append(ind)
    #print("q"*5,fits)

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)
    # the first sorted
    for i in range(0,len(fits)-1):
        fit_i = fits[i]
        for fit_j in fits[i + 1:]:
            if my_dominates(fit_i,fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif my_dominates(fit_j,fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    #print("z"*5,current_front)

    fronts = [[]]

    # sava the first pareto
    for fit in current_front:
        fronts[-1].append(fit)
    pareto_sorted = len(fronts[-1])

    # continue the remain sorted
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(fit_d)
                        fronts[-1].append(fit_d)
            current_front = next_front[:]
            next_front = []

    return fronts

def find_best_individual(pareto_first_front,pset):

    distances = []
    for ind in pareto_first_front:
        value = ind.fitness.values
        dis = math.sqrt(math.pow(value[0],2)+math.pow((value[1]-1),2))
        distances.append(dis)

    c = min(distances)
    min_list = []
    #看最优目标是否是唯一个体
    if distances.count(c) > 1:
        first_pos = 0
        for i in range(distances.count(c)):
            new_list = distances[first_pos:]
            select_index = new_list.index(c) + first_pos
            first_pos = select_index + 1
            k = len(count_selected_feat(pareto_first_front[select_index],pset))
            leaf_num = count_leaf_nodes(pareto_first_front[select_index])
            min_list.append((select_index,k,leaf_num))
        min_list.sort(key=lambda a:(a[1],a[2]))
        ev_min_dis = min_list[0]
        index = ev_min_dis[0]
        return pareto_first_front[index]
    else:
        index = distances.index(c)
        return pareto_first_front[index]
    

def find_individual(pareto_first_front,pset):
    distances = []
    for ind in pareto_first_front:
        value = ind.fitness.values
        dis = math.sqrt(math.pow(1-value[0],2)+math.pow((1-value[1]),2))
        distances.append(dis)
    c = min(distances)
    min_list = []
    #看最优目标是否是唯一个体
    if distances.count(c) > 1:
        first_pos = 0
        for i in range(distances.count(c)):
            new_list = distances[first_pos:]
            select_index = new_list.index(c) + first_pos
            first_pos = select_index + 1
            k = len(count_selected_feat(pareto_first_front[select_index],pset))
            leaf_num = count_leaf_nodes(pareto_first_front[select_index])
            min_list.append((select_index,k,leaf_num))
        min_list.sort(key=lambda a:(a[1],a[2]))
        ev_min_dis = min_list[0]
        index = ev_min_dis[0]
        return pareto_first_front[index]
    else:
        index = distances.index(c)
        return pareto_first_front[index]
    
    
def point_line_distance(k, b, point):
    c = k * point[0] - point[1] + b
    a = k * k + 1
    dis = abs(c) / math.sqrt(a)
    return dis


def find_min_distance(k, b, pareto_first_front, pset):
    dis_list = []

    for ind in pareto_first_front:
        i = point_line_distance(k, b, ind.fitness.values)
        dis_list.append(i)

    min_distance = min(dis_list)
    if dis_list.count(min_distance) > 1:
        min_list = []
        first_pos = 0
        for i in range(dis_list.count(min_distance)):
            new_list = dis_list[first_pos:]
            select_index = new_list.index(min_distance) + first_pos
            first_pos = select_index + 1
            feat_num = len(count_selected_feat(pareto_first_front[select_index], pset))
            leaf_num = count_leaf_nodes(pareto_first_front[select_index])
            min_list.append((select_index, feat_num,leaf_num))

        min_list.sort(key=lambda a: (a[1],a[2]))
        ev_min_dis = min_list[0]
        index = ev_min_dis[0]
        return pareto_first_front[index]
    else:
        index = dis_list.index(min_distance)
        return pareto_first_front[index]
#计算两点之间的斜率
def two_point_slope(first_point, second_point):
    two_fpr = first_point[0]-second_point[0]
    two_tpr = first_point[1]-second_point[1]
    if two_fpr == 0:
        return -10000
    else:
        k = two_tpr/two_fpr
        return k

#找到斜率之间的集成分类器
def find_best_ensembles (pareto_first_front,min_slope,max_slope):
    selected_best_ensembles = []
    perfect_point = (0, 1)

    for ind in pareto_first_front:
        current_point = ind.fitness.values
        k = two_point_slope(perfect_point,current_point)
        if(k > max_slope):
            break
        else:
            if(k >= min_slope):
                selected_best_ensembles.append(ind)

    return selected_best_ensembles

def simple_average_ensemble(ensembles,all_datas,toolbox):
    complie_list = []
    final_output = []
    for ind in ensembles:
        ind = toolbox.compile(ind)
        complie_list.append(ind)
    for data in all_datas:
        temp = []
        for individual in complie_list:
            k = individual(data[:-1])
            temp.append(k)
        z = sum(temp)/len(temp)
        final_output.append(z)
    return final_output
def my_class(individual,data):
    if individual(data[:-1]) >= 0:
        return 1.0
    else:
        return 0.0

def vote_ensemble(ensembles,all_datas,toolbox):
    final_list = []
    complie_list = []
    for ind in ensembles:
        ind = toolbox.compile(ind)
        complie_list.append(ind)
    for data in all_datas:
        temp = []
        for individual in complie_list:
            k = my_class(individual,data)
            temp.append(k)
        labels = set(temp)
        # print(labels)
        z = []
        for label in labels:
            a = temp.count(label)
            print(a)
            z.append(a)
        z.sort(reverse=True)
        final_list.append(z[0])
    print(final_list)
    print(len(final_list))
    return final_list

def select_mininze_feature(pop,pset):
    feature_list = []
    for ind in pop:
        k = len(count_selected_feat(ind,pset))
        leaf_num = count_leaf_nodes(ind)
        feature_list.append((k,leaf_num))
    min1 = min(feature_list,key=lambda v: (v[0], v[1]))
    index = feature_list.index(min1)
    return pop[index]

def find_unrepeat_ensembles(pareto_first_front,pset):
    final_list = []
    fitness_list = []
    for ind in pareto_first_front:
        fitness_list.append(ind.fitness.values)
    pre_point = 0
    temp = []
    for index, ind in enumerate(fitness_list):
        # print(ind)
        if index != 0:
            if temp[-1] == ind:
                temp.append(ind)
            else:
                new_list = pareto_first_front[pre_point:index]
                k = select_mininze_feature(new_list,pset)
                pre_point = index
                final_list.append(k)
                temp.append(ind)
        else:
            temp.append(ind)
    #处理最后一段
    new_list = pareto_first_front[pre_point:]
    k = select_mininze_feature(new_list, pset)
    final_list.append(k)

    return final_list


if __name__ == '__main__':
    a = (10, 5)
    k = 1
    b = 1
    z = point_line_distance(k,b,a)

    print(z)




