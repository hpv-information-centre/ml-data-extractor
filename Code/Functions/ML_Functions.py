from Functions.General_Functions import isNaN, getNContext, pad_array, candidateClsf
from Method_Handler import MethodHandler
import nltk
import math
import sklearn as sk
import numpy as np
import pandas as pd
import statistics as stats
import os
from tempfile import mkdtemp
from joblib import load, dump


pd.options.mode.chained_assignment = None  # default='warn'

def preprocessBy(method='CV-TFIDF', *args):
    return preprocessHandler.execMethod(method, None, *args)

def classifyWith(method='CV-SVM', *args):
    return classifHandler.execMethod(method, None, *args)

def CV_TFIDF(data, k):
    test_list, train_list = kSplit(data, k)
    pre_test = []
    pre_train = []
    IDF_list = []
    for i,t in enumerate(train_list):
        IDF = getIDF(t)
        if not os.path.isdir('Models'):
            os.mkdir('Models')
        path = os.path.abspath('Models')
        filename = os.path.join(path, 'IDF') + '_' + str(i) + '.joblib'
        dump(IDF, filename, compress=1)
        IDF_list.append(IDF)
        pre_train.append(transformSet(t, IDF))
        pre_test.append(transformSet(test_list[i], IDF))
    return IDF_list, pre_train, pre_test

def kSplit(data, k):
    test = []
    train = []
    l = len(data)
    n = math.floor(l/k)
    r = l % k
    index = 0
    n += 1
    for i in range(k):
        if r == 0: 
            r = -1
            n -= 1
        test_split = data.loc[index:(index+n-1),:]
        test_split = test_split.reset_index(drop=True)
        train_split = data.loc[:index-1,:]
        train_split = train_split.append(data.loc[(index+n):,:], ignore_index=True)
        train_split = train_split.reset_index(drop=True)
        index += n
        if r > 0: r -= 1
        test.append(test_split)
        train.append(train_split)
    return test, train
        

def filterDotEmpty(token):
    return False if token in ['.', '','(',')','<','>',',',':',';','!','?','[',']','{','}','-','/','\\'] else True  

def splitToken(tk, splitList):
    stk = tk
    for symbol in splitList:
        if symbol in stk:
            stk = stk.split(symbol)[1]
    if isNaN(stk): stk = tk
    return stk

def generateNumbersDataFrame(data):
    abstractList = data['Abstract'].to_list()
    for i, abstract in enumerate(abstractList):
        new_sentence = []
        N_list = []
        N_sentences_list = []
        N_close_words_list = []
        sentences = nltk.sent_tokenize(abstract)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tokens = filterTokens(tokens)
            for j,token in enumerate(tokens):
                if not isNaN(token) and "." not in token and (j+1 < len(tokens) and "%" not in tokens[j+1]) and float(token) > 2: #token must be an integer number and the next token cannot be a % sign
                    N_list.append(token)
                    N_sentences_list.append(filterNumbers(tokens))
                    N_close_words_list.append(filterNumbers(getNContext(tokens, j, 3)))
        partial_data = pd.DataFrame(data={'PMID': [data['PMID'][i]]*len(N_list), 'N': N_list, 'N sentence words': N_sentences_list, 'N close words': N_close_words_list})
        full_data = full_data.append(partial_data, ignore_index=True, sort=True) if 'full_data' in locals() else partial_data
    full_data = candidateClsf(full_data, data)
    return full_data

def filterTokens(tokens):
    new_sentence = []
    tokens = list(filter(filterDotEmpty, tokens))
    for token in tokens:
        token = token.replace(",", "") #Take out commas from numbers (english format)
        token = splitToken(token, ['/', '-'])
        new_sentence.append(token)
    return new_sentence

def getIDF(train_set):
    #vocab = getVocab(train_set['N close words'])
    vect = sk.feature_extraction.text.TfidfVectorizer(max_features=None, use_idf=True, vocabulary=None, min_df=0.03)
    words = []
    for i, doc in enumerate(train_set['N sentence words']):
        if train_set['Is Candidate'][i] == 1:
            words.append(' '.join(train_set['N close words'][i]))
    IDF = vect.fit(words)
    return IDF
    
def getVocab(column):
    vocab = []
    for word_bag in column:
        for word in word_bag:
            if word not in vocab and isNaN(word):
                vocab.append(word)
    return vocab

def filterNumbers(doc):
    d = []
    for token in doc:
        if isNaN(token):
            d.append(token)
    return d

def transformSet(dataset, IDF):
    close_words = []
    for word_list in dataset['N close words']:
        close_words.append(' '.join(word_list))
    X = IDF.transform(close_words)
    weights = []
    means = []
    for row in X:
        weights.append(row.toarray())
        means.append(stats.mean(row.toarray()[0]))
    IDF_words = IDF.vocabulary_.keys()
    dataset['N close word weights'] = weights
    dataset['Weight means'] = means
    dataset = createWordColumns(dataset, IDF_words)
    return dataset

def createWordColumns(df, words):
    for i in df.index:
        l = len(df.index)
        weights = df['N close word weights'].iloc[i][0]
        for j, w in enumerate(words):
            if w not in df.columns:
                df[w] = pd.to_numeric([0.0]*l, downcast='float')
            if w in df.columns:
                df[w][i] = weights[j]
    return df

def SVM(train, test, index, c, kernel, degree, gamma, prob):
    classifier = sk.svm.SVC(C=c,kernel=kernel, degree=degree, gamma=gamma, probability=prob)
    classifier.fit(np.matrix(train.iloc[:,6:]), np.array(train['Is Candidate']))
    class_results = classifier.predict(np.matrix(test.iloc[:,6:]))
    class_prob = classifier.predict_proba(np.matrix(test.iloc[:,6:]))
    if not os.path.isdir('Models'):
        os.mkdir('Models')
    path = os.path.abspath('Models')
    filename = os.path.join(path, 'SVM')
    if index != None and isinstance(index, int):
        dump(classifier, filename + '_' + str(index) + '.joblib', compress=1)
    else:
        dump(classifier, filename + '.joblib', compress=1)
    return class_results, class_prob

def CV_SVM(train_list, test_list):
    true_class = []
    predicted_class = []
    true_class_probs = []
    for i, t in enumerate(train_list):
        predictions, probs = classifyWith('SVM', t, test_list[i], i, 1, 'rbf', 3, 'scale', True)
        true_class.extend(test_list[i]['Is Candidate'])
        predicted_class.extend(predictions)
        true_class_probs.extend(probs[:,1])
    return true_class, predicted_class, true_class_probs


#PREPROCESS METHODS
preProcessMethods = {
    "CV-TFIDF": CV_TFIDF
}

classifMethodDicc = {
    "SVM": SVM,
    "CV-SVM": CV_SVM
} 

preprocessHandler = MethodHandler(preProcessMethods)
classifHandler = MethodHandler(classifMethodDicc)