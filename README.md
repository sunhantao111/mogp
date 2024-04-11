# mogp
使用NSGA-II算法优化三个目标recall，precision，complexity
## 需要安装的包
```
pip install scikit-learn deap matplotlib
```
## 需要更改的文件
将已安装好的deap库中的gp.py和base.py替换为mogp文件夹下的gp.py和base.py

在deap库中的algorithms.py文件中添加nsga2方法：
```python
# 自定义NSGA2算法
def eaNSGA2(population, toolbox, cxpb, mutpb, ngen, npop, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # 为初始种群进行评价，得到适应度，这里先有一个适应度才能进行快速非支配排序
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fitness in zip(population, fitnesses):
        ind.fitness.values = fitness

    # 用生成的种群更新名人堂，名人堂存储的是非支配解（rank1中的解，pareto最优解）
    if halloffame is not None:
        halloffame.update(population)

    # 将当前生成的统计信息追加到日志中
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, **record)
    if verbose:
        print(logbook.stream)

    # 开始迭代
    # 第一代与第二代之后的代数操作不同
    # 快速非支配排序，得到不同前沿的pareto层集合fronts
    fronts = tools.emo.sortNondominated(population, k=npop, first_front_only=False)
    for i, front in enumerate(fronts):     # 使用枚举循环得到各层的标号与pareto解
        for ind in front:
            ind.fitness.values = i + 1,    # 将个体的适应度设定为pareto解的前沿次序

    # 进行选择 锦标赛选择法
    offspring_sl = toolbox.selectGen1(population, npop)
    # 只做一次交叉与变异操作
    offspring = varAnd(offspring_sl, toolbox, cxpb, mutpb)

    # 从第二代开始循环
    for gen in range(1, ngen):
        # 对后代（offspring）进行评价，得到适应度
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fitness in zip(offspring, fitnesses):
            ind.fitness.values = fitness

        # 将父代与子代结合成一个大种群
        combinedPop = population + offspring

        # 对该大种群进行适应度计算
        fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
        for ind, fitness in zip(combinedPop, fitnesses):
            ind.fitness.values = fitness

        print("-" * 50 + "第{}代".format(gen) + "-" * 50)

        # 对该大种群进行快速非支配排序
        fronts = tools.emo.sortNondominated(combinedPop, len(combinedPop))
        # 基于拥挤度实现精英保存策略
        population = tools.selNSGA2(combinedPop, k=npop, nd='standard')
        # 选择
        matingpool = toolbox.select(population, npop)
        #print(rept_rate(population))

        # 用生成的种群更新名人堂
        if halloffame is not None:
            halloffame.update(population)

        # 将当前生成的统计信息追加到日志中
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print(logbook.stream)
        # 交叉变异
        offspring = varAnd(matingpool, toolbox, cxpb, mutpb)

    return population,halloffame
```
在gp_nsga2_main.py中需要将前两行代码更改为自己的路径
```python
resultspath = 'D:\\fpr-tpr\\gp_nsga2'
dir_name = 'D:\\fpr-tpr\\gp_nsga2\\uci\\biodata2'
```
# 运行
python gp_nsga2_main.py
