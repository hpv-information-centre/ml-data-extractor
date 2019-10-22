from __future__ import division
from Functions.General_Functions import importFrom, pubmedInfoDownload, saveFile
from Functions.ML_Functions import preprocessBy, generateNumbersDataFrame, classifyWith
from Method_Handler import MethodHandler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from joblib import load, dump
import os
import openpyxl



############## FIRST DATA DOWNLOAD ##############
# data = importFrom('xl', 'Data/infocentre_data.xlsx')
# data = pubmedInfoDownload(data)
# saveFile(data, './Data/AbstractDF.pkl')
############## END FIRST DATA DOWNLOAD ##############

data = importFrom('pkl', './Data/AbstractDF.pkl')

if isinstance(data, pd.DataFrame):
    ############## PREPROCESS TRAINING ##############
    data = generateNumbersDataFrame(data)
    saveFile(data, './Data/NumbersDF.pkl') #save df to excel
    #CUSTOM CROSS-VALIDATION
    k = 10
    IDF_list, pre_train, pre_test = preprocessBy('CV-TFIDF', data, k)
    data = pre_train[0].append(pre_test[0])
    for i,t in enumerate(pre_train):
        saveFile(pre_test[i], './Data/Pre_testv5_' + str(i) + '.pkl')
        saveFile(t, './Data/Pre_trainv5_' + str(i) + '.pkl')
    ############# END PREPROCESS TRAINING ##############
    

    ############## LOAD PREPROCESS DF ##############
    pre_train = []
    pre_test = []
    IDF_list = []
    k = 10
    path = os.path.abspath('Models')
    for i in range(k):
        pre_train.append(importFrom('pkl', './Data/Pre_trainv3_' + str(i) + '.pkl'))
        pre_test.append(importFrom('pkl', './Data/Pre_testv3_' + str(i) + '.pkl'))
        filename = os.path.join(path, 'IDFv3') + '_' + str(i) + '.joblib'
        IDF_list.append(load(filename))
    ############## END LOAD PREPROCESS DF ##############


#     ############## CORRELATION GRAPHS ##############
#     # data = pre_test[0].append(pre_train[0])
#     # corr_data = data[IDF_list[0].vocabulary_.keys()]
#     # corr_matrix = corr_data.corr()
#     # fig, ax = plt.subplots(figsize=(20, 12))
#     # a = sns.heatmap(corr_matrix, vmax=1.0, square=True, ax=ax)
#     # b = sns.pairplot(corr_data)
#     # plt.show()
#     ############## END CORRELATION GRAPHS ##############

    ############## MODEL TRAINING ##############
    true_class, predicted_class, true_class_probs = classifyWith('CV-SVM', pre_train, pre_test)
    ############## END MODEL TRAINING ##############

    ############## GRAPHICS ##############
    print('--------------------MODEL GENERAL METRICS---------------------')
    #confusion matrix
    cmatrix = metrics.confusion_matrix(true_class, predicted_class)
    print('Confusion Matrix:\n', cmatrix)
    #fscore
    fscore = metrics.f1_score(true_class, predicted_class)
    print('F-Score: ', fscore)
    #cappa kohen
    kappa = metrics.cohen_kappa_score(true_class, predicted_class)
    print('Kappa Cohen: ', kappa)

    #ROC
    print('\n----------ROC----------')
    fpr, tpr, thresholds = metrics.roc_curve(true_class, true_class_probs)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC: ', roc_auc)
    #Perfect
    plt.plot([0,1], [1,1], linestyle='--')
    #Worst
    plt.plot([0,1], [0,1], linestyle='--')
    #Our model
    plt.plot(fpr, tpr)
    plt.show()

    ####################################### Precision-Recall ######################################
    print('\n----------Precision/Recall----------')
    precision, recall, thresholds2 = metrics.precision_recall_curve(true_class, true_class_probs)
    #AUC
    auc = metrics.auc(recall, precision)
    print('AUC: ', auc)
    #no skill model
    trues = true_class.count(1)
    predicted_trues = predicted_class.count(1)
    total = len(true_class)
    apriori_prob = trues/total
    plt.plot([0,1], [apriori_prob,apriori_prob], marker='.')
    #our model
    plt.plot(recall, precision)
    plt.show()
    ############## END GRAPHICS ##############

else:
    print('Method does NOT exist.')










