from __future__ import division
import requests
from Bio import Entrez
import nltk
import pandas as pd
from joblib import load, dump
import sklearn as sk
import statistics as stats
import numpy as np
import xlrd
import matplotlib.pyplot as plt 



def pubmedAbstractDownload(art_id):
    Entrez.email = 'marcarmenter@hotmail.com'
    query = Entrez.efetch(db='pubmed', retmode='xml', report='abstract', id=art_id)
    try:
        res = getAbstractFromEntrezQuery(Entrez.read(query))
    except:
        return "No article found."
    return res

def getAbstractFromEntrezQuery(EntrezQuery):
    citation = EntrezQuery['PubmedArticle'][0]['MedlineCitation']
    abstract = ' '.join(citation['Article']['Abstract']['AbstractText'])       
    return abstract

def getCandidates(abstract):
    new_abstract = ""
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
        for t in tokens:
            if not isNaN(t) and "." not in t and float(t) > 2: #token must be an integer number
                new_abstract += "<b>" + t + "</b> "
            elif t in [",",".",";",":", ")", "]", "}", "%"]:
                new_abstract = new_abstract[:-1]
                new_abstract += str(t) + " "
            elif t in ["[","{","("]:
                new_abstract += str(t)
            elif "/" in t:
                s = t.split("/")
                if not isNaN(s[-1]):
                    s[-1] = "<b>" + s[-1] + "</b> "
                new_abstract += "/".join(s)
            else:
                new_abstract += str(t) + " "
                
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
        for j,token in enumerate(pos_tokens):
            if not isNaN(token[0]) and "." not in token[0] and float(token[0]) > 2: #token must be an integer number
                N_list.append(token[0])
                N_sentences_list.append(filterNumbers(lemma_tokens))
                #words, distances = getNContext(tokens, j, 3) method 1
                words, distances = getNPfromNumber(sentence_tree, token[0], j) #method 2
                N_close_words_list.append(words)
                N_close_words_distances_list.append(distances)
                N_isFraction_list.append(isFraction_list[j])
    partial_data = pd.DataFrame(data={'N': N_list, 'N sentence words': N_sentences_list, 'N close words': N_close_words_list, 'N close words distances': N_close_words_distances_list, 'Is fraction': N_isFraction_list})
    saveFile(partial_data, './abstract_demo_app/Data/Preprocess.pkl')
    return new_abstract

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

def isNaN(val):
    val = val.replace(",", "")
    try:
        val = float(val)
    except:
        pass
        return True
    return val != val

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

def filterNumbers(doc):
    d = []
    for token in doc:
        if isNaN(token):
            d.append(token)
    return d

def filterNumbers_MaintainDistances(doc, distances):
    d = []
    c = []
    for i, token in enumerate(doc):
        if isNaN(token):
            d.append(token)
            c.append(distances[i])
    return d, c

def saveFile(data, path):
    try:
        data.to_pickle(path)
    except:
        print('Unable to save file')

def getCandidatePredictions(text):
    test = pd.read_pickle('./abstract_demo_app/Data/Preprocess.pkl')

    # train = pd.read_pickle("./abstract_demo_app/Data/NumbersDF.pkl")
    # IDF = getMeanIDF(train)
    # train = transformSet(train, IDF)
    # SVM(train, 2, 'rbf',  'scale', True)

    IDF = load("./abstract_demo_app/Data/IDF_mean.joblib")
    test = transformSet(test, IDF)
    classifier = load('./abstract_demo_app/Data/SVM.joblib')
    test_set = test.iloc[:,6:]
    test_set['Is fraction'] = test['Is fraction']
    class_results = classifier.predict(np.matrix(test_set))
    class_prob = classifier.predict_proba(np.matrix(test_set))
    probs = class_prob[:,1]
    print(probs)
    # fpr, tpr, thresholds = sk.metrics.roc_curve([0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0], probs)
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    # roc_auc = sk.metrics.auc(fpr, tpr)
    # print('AUC: ', roc_auc)
    # #Perfect
    # plt.plot([0,1], [1,1], linestyle='--')
    # #Worst
    # plt.plot([0,1], [0,1], linestyle='--')
    # #Our model
    # plt.plot(fpr, tpr)
    # plt.show()

    for prediction in probs:
        if prediction >= 0.02633436: text = text.replace("<b>", "<b style='color:green'>", 1)
        else: text = text.replace("<b>", "<b style='color:red'>", 1)
    return text

def getMeanIDF(train):
    # k = 10
    # vocab_t = vocab = {}
    # for i in range(k):
    #     IDF = load("./abstract_demo_app/Data/IDFv5_" + str(i) + ".joblib")
    #     for word, weight in IDF.vocabulary_.items():
    #         if word not in vocab_t.keys():
    #             vocab_t[word] = (weight, 1)
    #         else:
    #             vocab_t[word] = (vocab_t[word][0] + weight, vocab_t[word][1] + 1)
    # for paraula, pes in vocab_t.items():
    #     vocab[paraula] = vocab_t[paraula][0]/vocab_t[paraula][1]
    vect = sk.feature_extraction.text.TfidfVectorizer(max_features=None, use_idf=True, vocabulary=None, min_df=0.01)
    words = []
    for i, doc in enumerate(train['N sentence words']):
        if train['Is Candidate'][i] == 1:
            words.append(' '.join(train['N close words'][i]))
    IDF = vect.fit(words)
    dump(IDF, "./abstract_demo_app/Data/IDF_mean.joblib", compress=1)
    return IDF

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
            df[w][i] = weights[j]
            if w in df['N close words'].iloc[i]:
                windex = df['N close words'].iloc[i].index(w)
                df[str(w + ' dist')][i] = distances[windex]
    return df

def SVM(train, c, kernel,  gamma, prob):
    classifier = sk.svm.SVC(C=c,kernel=kernel, gamma=gamma, class_weight='balanced', probability=prob)
    train_set = train.iloc[:,8:]
    train_set['Is fraction'] = train['Is fraction']
    classifier.fit(np.matrix(train_set), train['Is Candidate'])
    dump(classifier, './abstract_demo_app/Data/SVM.joblib', compress=1)