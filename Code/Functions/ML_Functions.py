from Functions.General_Functions import isNaN, getNContext, pad_array
import sklearn as sk
import numpy as np
import pandas as pd
import nltk

pd.options.mode.chained_assignment = None  # default='warn'

def bag_of_words(data, ids, content, lowcase, ngram_rang):
    _tdfidf_contexts = pd.DataFrame(columns={'N', 'N_context', 'N_context_words'})
    vect = sk.feature_extraction.text.TfidfVectorizer(min_df=5, max_features=None, use_idf=True)
    X = vect.fit(content)
    max_context_length = 0
    #Tokenize abstracts
    for i, atcl in enumerate(content):
        atcl = nltk.word_tokenize(atcl)
        atcl_ids = []
        atcl_Ns = []
        N_context_list = []
        N_context_words_list = []
        is_ico_indicator_list = []
        
        #Find all N's and add the new info
        for j, tk in enumerate(atcl):            
            if not isNaN(tk) and "." not in tk and (  "%" not in atcl[j+1] and j < len(atcl) ): #tk must be an integer number and the next token cannot be a % sign
                context = []
                tk = tk.replace(",", "")
                #Get context
                context.append(getNContext(atcl, j, 20))
                C = X.transform(context) #@transform only accepts list items
                #C.sort_indices()
                ##########
                
                if len(C.data) > max_context_length:
                    max_context_length = len(C.data)

                #Sort C by index
                c = pd.DataFrame(data={'values': C.data, 'words': X.inverse_transform(C)[0], 'indices': C.indices}) #@inverse_transform returns a list of lists
                #c = c.sort_values(by='indices',ascending=False)
                ###########

                atcl_ids.append(ids[i])
                atcl_Ns.append(tk)
                N_context_list.append(c['values'].to_list())
                N_context_words_list.append(c['words'].values)
                if data.loc[i,'n_ic'] == float(tk): 
                    is_ico_indicator_list.append(1)
                else: 
                    is_ico_indicator_list.append(0)
                ############

        #Include N contexts to dataframe        
        context_df = pd.DataFrame(data={'atcl_id': atcl_ids, 'N': atcl_Ns, 'N_context': N_context_list, 'N_context_words': N_context_words_list, 'is_candidate': is_ico_indicator_list})
        _tdfidf_contexts = _tdfidf_contexts.append(context_df, ignore_index=True, sort=True)        
        ################################
    
    for k,a in enumerate(_tdfidf_contexts['N_context']):
        _tdfidf_contexts['N_context'][k] = pad_array(a, max_context_length)

    _tdfidf_contexts['is_candidate'] = _tdfidf_contexts['is_candidate'].astype('int')
    ########################



    #Get global text statistics
    X = vect.fit_transform(content)
    #X.sort_indices()
    N = vect.get_feature_names() #Most relevant words
    Y = vect.inverse_transform(X)
    values = X.data
    indices = X.indices
    _names = []
    for text in Y:
        for word in text:
            _names.append(word)

    d = {'indices': indices, 'values': values, 'names': _names}
    _tdfidf = pd.DataFrame(data=d) #TFIDF dataframe
    _tdfidf = _tdfidf.sort_values(by='values',ascending=False)

    _sorted_names = _tdfidf['names'].unique() #sorted by tf desc value, same as variable N but sorted
    ########################

    #CLASSIFICATION
    _classifier = sk.svm.SVC(C=1,kernel='rbf', degree=3, gamma='scale', probability=True)
    #separate df in 80-20%
    split = round(len(_tdfidf_contexts)*0.8)
    train = _tdfidf_contexts.loc[:split, :]
    test = _tdfidf_contexts.loc[split+1:, :]    
    ##########    
    _classifier.fit(np.matrix(train['N_context'].to_list()), np.array(train['is_candidate']))    
    _class_results = _classifier.predict(np.matrix(test['N_context'].to_list()))
    _class_prob = _classifier.predict_proba(np.matrix(test['N_context'].to_list()))    
    ########################
    
    return {
        'globalDF': _tdfidf,
        'bow': _sorted_names,
        'contextDF': _tdfidf_contexts,
        'classifier': _classifier,
        'class_results': _class_results,
        'class_prob': _class_prob
    }

# def knn(arg, arg2):
#     print('2')

