from __future__ import division
from Functions.General_Functions import importFrom, pubmedInfoDownload, saveFile
from Functions.ML_Functions import preprocessBy, generateNumbersDataFrame, classifyWith
from Method_Handler import MethodHandler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

############## FIRST DATA DOWNLOAD ##############
# data = importFrom('xl', 'Data/infocentre_data.xlsx')
# data = pubmedInfoDownload(data)
# saveFile(data, './Data/AbstractDF.pkl')
############## END FIRST DATA DOWNLOAD ##############

data = importFrom('pkl', './Data/AbstractDF.pkl')
if isinstance(data, pd.DataFrame):
    ############## PREPROCESS TRAINING ##############
    data = generateNumbersDataFrame(data)
    #CUSTOM CROSS-VALIDATION
    k = 10
    IDF_list, pre_train, pre_test = preprocessBy('CV-TFIDF', data, k)
    ############## END PREPROCESS TRAINING ##############
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

    #######################################Precision-Recall ######################################
    print('\n----------Precision/Recall----------')
    precision, recall, thresholds2 = metrics.precision_recall_curve(true_class, true_class_probs)
    #AUC
    auc = metrics.auc(recall, precision)
    print('AUC: ', auc)
    #no skill model
    plt.plot([0,1], [0.5,0.5], marker='.')
    #our model
    plt.plot(recall, precision)
    plt.show()
    ############## END GRAPHICS ##############

else:
    print('Method does NOT exist.')




# keywordsICO = ['women', 'participants', 'subjects', 'attendants', 'controls']





#IMPLEMENTAR CLASSIFICACIO AMB NN KERAS/TENSORFLOW











