import pandas as pd

import numpy as np
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import random
from read_data import read_arff
from sklearn.metrics import roc_auc_score

knn = KNeighborsClassifier()
svm = svm.SVC()
lr = LogisticRegression()
nb = GaussianNB()
mlp = MLPClassifier()
dt_gini = tree.DecisionTreeClassifier(criterion='gini')
dt_entropy = tree.DecisionTreeClassifier(criterion='entropy')
#acc
acc_knn = []
acc_svm = []
acc_lr = []
acc_nb = []
acc_mlp = []
acc_dt_gini = []
acc_dt_entropy = []
#auc
auc_knn = []
auc_svm = []
auc_lr = []
auc_nb = []
auc_mlp = []
auc_dt_gini = []
auc_dt_entropy = []

# # dir_name = "../datasets/csv/"
# GSES = ['GSE14728','GSE42408', 'GSE46205', 'GSE76613', 'GSE145709']
# resultspath = '../JILU/'

dir_name = 'F:\\three objective\\gp_nsga2\\uci\\biodata2'
#GSES = ["ionosphere","wdbc","sonar","Leukemia","HillValley","Colon","liver-disorders"]
GSES = ['GSE14728','GSE30464','GSE42408', 'GSE46205','GSE76613', 'GSE145709']
#GSES = ["GSE65046","GSE71723","GSE98455"]
#GSES = ['HillValley']
resultspath = 'F:\\three objective\\gp_nsga2'


# dir_name = 'E:\\project_FRFS\\dataset&results\\revise_dataset_selected\\'
# #GSES = ['selectedgenes_1','selectedgenes_2']
# resultspath = 'E:\\project_FRFS\\dataset&results\\revise_dataset_selected\\JILU'

def main():
    with open(resultspath + 'gene_auc.txt', 'a') as f:
        f.write('KNN\t\t\tSVM\t\t\tLR\t\t\tnb\t\t\tmlp\t\t\tdt_gini\t\t\tdt_entropy\t\t\tDATASET\n')
        for GSE in GSES:
            #dataset = pd.read_csv(dir_name + GSE + '.csv')
            dataset ,flag = read_arff(dir_name, GSE)
            dataset = np.array(dataset)
            X = dataset[:, :-1]
            y = dataset[:, -1]
            b = []
            for i in range(40):
                a = random.randint(0,42)
                b.append(a)
            for i in range(40):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3,
                                                                    random_state=b[i], stratify=y)
                #print(X_train)
                knn.fit(X_train, y_train)
                # acc_knn.append(knn.score(X_test, y_test))
                auc_knn.append(roc_auc_score(y_test,knn.predict_proba(X_test)[:, 1]))

                svm.fit(X_train, y_train)
               # acc_svm.append(svm.score(X_test, y_test))
                auc_svm.append(roc_auc_score(y_test, svm.decision_function(X_test)))

                lr.fit(X_train, y_train)
                #acc_lr.append(lr.score(X_test, y_test))
                auc_lr.append(roc_auc_score(y_test, lr.decision_function(X_test)))

                nb.fit(X_train, y_train)
                #acc_nb.append(nb.score(X_test, y_test))
                auc_nb.append(roc_auc_score(y_test, nb.predict_proba(X_test)[:, 1]))

                mlp.fit(X_train, y_train)
               # acc_mlp.append(mlp.score(X_test, y_test))
                auc_mlp.append(roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]))

                dt_gini.fit(X_train, y_train)
               # acc_dt_gini.append(dt_gini.score(X_test, y_test))
                auc_dt_gini.append(roc_auc_score(y_test, dt_gini.predict_proba(X_test)[:, 1]))

                dt_entropy.fit(X_train, y_train)
               # acc_dt_entropy.append(dt_entropy.score(X_test, y_test))
                auc_dt_entropy.append(roc_auc_score(y_test, dt_entropy.predict_proba(X_test)[:, 1]))

            # f.write(GSE + '\t' + str(round(mean(acc_knn) * 100, 3)) + '\t'
            #         + str(round(mean(acc_svm) * 100, 3)) + '\t'
            #         + str(round(mean(acc_lr) * 100, 3)) + '\t'
            #         + str(round(mean(acc_nb) * 100, 3)) + '\t'
            #         + str(round(mean(acc_mlp) * 100, 3)) + '\t'
            #         + str(round(mean(acc_dt_gini) * 100, 3)) + '\t'
            #         + str(round(mean(acc_dt_entropy) * 100, 3)) + '\n')

            f.write(str(round(mean(auc_knn) * 100, 4)) + '\t\t'
                    + str(round(mean(auc_svm) * 100, 4)) + '\t\t'
                    + str(round(mean(auc_lr) * 100, 4)) + '\t\t'
                    + str(round(mean(auc_nb) * 100, 4)) + '\t\t'
                    + str(round(mean(auc_mlp) * 100, 4)) + '\t\t'
                    + str(round(mean(auc_dt_gini) * 100, 4)) + '\t\t\t'
                    + str(round(mean(auc_dt_entropy) * 100, 4)) + '\t\t\t'+GSE +'\n')
    f.close()


if __name__ == '__main__':
    main()
