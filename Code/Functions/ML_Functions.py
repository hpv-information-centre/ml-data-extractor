from Functions.General_Functions import isNaN, getNContext, pad_array, candidateClsf, filterNumbers, filterNumbers_MaintainDistances
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
        filename = os.path.join(path, 'IDFv6') + '_' + str(i) + '.joblib'
        dump(IDF, filename, compress=1)
        IDF_list.append(IDF)
        pre_train.append(transformSet(t, IDF))
        pre_test.append(transformSet(test_list[i], IDF))
    return IDF_list, pre_train, pre_test

def kSplit(data, k):
    data = sk.utils.shuffle(data) #randomly shuffled
    data = data.reset_index(drop=True)
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
    isFraction = False
    for symbol in splitList:
        if symbol in stk:
            stk = stk.split(symbol)[1]
    if isNaN(stk): stk = tk
    elif '/' in tk: isFraction = True
    return stk, isFraction

def text2int(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    # ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    # ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for i,word in enumerate(textnum.split()):
        # if word in ordinal_words:
        #     scale, increment = (1, ordinal_words[word])
        #     current = current * scale + increment
        #     if scale > 100:
        #         result += current
        #         current = 0
        #     onnumber = True
        # else:
        #     for ending, replacement in ordinal_endings:
        #         if word.endswith(ending):
        #             word = "%s%s" % (word[:-len(ending)], replacement)

        if word not in numwords:
            if onnumber:
                if result == current == 0 and textnum.split()[i-1] == 'and':
                    curstring += "and "
                else:
                    curstring += repr(result + current) + " "
            curstring += word + " "
            result = current = 0
            onnumber = False
        else:
            scale, increment = numwords[word]

            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True

    if onnumber:
        curstring += repr(result + current)
    return curstring

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def generateNumbersDataFrame(data):
    abstractList = data['Abstract'].to_list()
    for i, abstract in enumerate(abstractList):
        new_sentence = []
        N_list = []
        N_sentences_list = []
        N_close_words_list = []
        N_close_words_distances_list = []
        N_isFraction_list = []
        abstract = text2int(abstract)
        sentences = nltk.sent_tokenize(abstract)
        for sentence in sentences:
            #tokenize
            tokens = nltk.word_tokenize(sentence)
            #lemmatize
            lemmatizer = nltk.stem.WordNetLemmatizer()
            lemma_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
            lemma_tokens, isFraction_list = filterTokens(lemma_tokens)
            #pos tagging
            pos_tokens = nltk.pos_tag(lemma_tokens)
            #chuncking
            grammar = r"""
                NP:                 # NP stage
                {(<DT>*)?(<CD>)*?(<RB>)?(<VBP>*)?(<JJ.*>*)?<NN.*>*(<VBP>*)?(<JJ.*>*)?(<NN.*>*)?}
                VP:
                {(<MD>)?(<TO>)?<VB.*>*(<RP>)?}
                """
            chunk_parser = nltk.RegexpParser(grammar)
            sentence_tree = chunk_parser.parse(pos_tokens)
            #sentence_tree.draw()
            for j,token in enumerate(pos_tokens):
                if not isNaN(token[0]) and "." not in token[0] and float(token[0]) > 2: #token must be an integer number
                    N_list.append(token[0])
                    N_sentences_list.append(filterNumbers(lemma_tokens))
                    #words, distances = getNContext(tokens, j, 3) method 1
                    words, distances = getNPfromNumber(sentence_tree, token[0], j) #method 2
                    N_close_words_list.append(words)
                    N_close_words_distances_list.append(distances)
                    N_isFraction_list.append(isFraction_list[j])
        partial_data = pd.DataFrame(data={'PMID': [data['PMID'][i]]*len(N_list), 'N': N_list, 'N sentence words': N_sentences_list, 'N close words': N_close_words_list, 'N close words distances': N_close_words_distances_list, 'Is fraction': N_isFraction_list})
        full_data = full_data.append(partial_data, ignore_index=True, sort=True) if 'full_data' in locals() else partial_data
    full_data = candidateClsf(full_data, data)
    return full_data

def getNPfromNumber(sent_tree, number, index):
    a = q = -1
    b = []
    for tree in sent_tree:
        if isinstance(tree, tuple):
            tree = [tree]
        for word in tree:
            a += 1
            if a == index:
                b = [w[0] for w in tree]
                c = [abs(index-q+i) for i,w in enumerate(tree)]
                return filterNumbers_MaintainDistances(b,c)
        q = a + 1 #indice inicio sub-arbol
    return []

def filterTokens(tokens):
    new_sentence = []
    isFraction_list = []
    tokens = list(filter(filterDotEmpty, tokens))
    for token in tokens:
        token = token.replace(",", "") #Take out commas from numbers (english format)
        token, isFraction = splitToken(token, ['/', '-'])
        new_sentence.append(token)
        isFraction_list.append(isFraction)
    return new_sentence, isFraction_list

def getIDF(train_set):
    #vocab = getVocab(train_set['N close words'])
    vect = sk.feature_extraction.text.TfidfVectorizer(max_features=None, use_idf=True, vocabulary=None, min_df=0.01)
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
    dataset['N close words weights'] = weights
    dataset['Weight means'] = means
    dataset = createWordColumns(dataset, IDF_words, True)
    return dataset

def createWordColumns(df, words, use_distances=False):
    for i in df.index:
        l = len(df.index)
        weights = df['N close words weights'].iloc[i][0]
        distances = df['N close words distances'].iloc[i]
        for j, w in enumerate(words):
            if w not in df.columns:
                df[w] = pd.to_numeric([0.0]*l, downcast='float')
                df[str(w + ' dist')] = pd.to_numeric([50.0]*l, downcast='integer') #entendiendo 50 como una distancia muy grande
            if w in df.columns:
                df[w][i] = weights[j]
            if w in df['N close words'].iloc[i]:
                windex = df['N close words'].iloc[i].index(w)
                df[str(w + ' dist')][i] = distances[windex]
    return df

def SVM(train, test, index, c, kernel,  gamma, prob):
    classifier = sk.svm.SVC(C=c,kernel=kernel, gamma=gamma, class_weight='balanced', probability=prob)
    train_set = train.iloc[:,8:]
    train_set['Is fraction'] = train['Is fraction']
    test_set = test.iloc[:,8:]
    test_set['Is fraction'] = test['Is fraction']
    classifier.fit(np.matrix(train_set), train['Is Candidate'])
    class_results = classifier.predict(np.matrix(test_set))
    class_prob = classifier.predict_proba(np.matrix(test_set))
    if not os.path.isdir('Models'):
        os.mkdir('Models')
    path = os.path.abspath('Models')
    filename = os.path.join(path, 'SVMv8')
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
        predictions, probs = classifyWith('SVM', t, test_list[i], i, 2, 'rbf', 'scale', True)
        true_class.extend(test_list[i]['Is Candidate'])
        predicted_class.extend(predictions)
        true_class_probs.extend(probs[:,1])
    return true_class, predicted_class, true_class_probs

def NN(train, test, index, *args):
    return 0

def CV_NN(train_list, test_list):
    true_class = []
    predicted_class = []
    true_class_probs = []
    for i, t in enumerate(train_list):
        a = 0
    return true_class, predicted_class, true_class_probs

#PREPROCESS METHODS
preProcessMethods = {
    "CV-TFIDF": CV_TFIDF
}

classifMethodDicc = {
    "SVM": SVM,
    "CV-SVM": CV_SVM,
    "NN": NN,
    "CV-NN": CV_NN
} 

preprocessHandler = MethodHandler(preProcessMethods)
classifHandler = MethodHandler(classifMethodDicc)