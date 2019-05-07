from Functions.General_Functions import xl_export
from Functions.ML_Functions import bag_of_words
import numpy as np
import nltk
import re
import math
import requests
from Bio import Entrez
import pandas as pd

#Data export from excel
_data = xl_export('Data/infocentre_data.xlsx')
#########################

#Entrez API calls
_entrezRoot = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
_esummary = 'esummary.fcgi?db=pubmed&retmode=xml&id='

_pmidsList = _data['pmid'].to_list()
_PMIDS = ','.join(_pmidsList)

Entrez.email = 'marcarmenter@hotmail.com'
_query = Entrez.efetch(db='pubmed', retmode='xml', report='abstract', id=_PMIDS) #_PMIDS, 24192311,23534783
_res = Entrez.read(_query)
#########################

#Extract Abstracts
_articleList = _res['PubmedArticle']
_abstracts = []
for atcl in _articleList:
    _abstracts.append(atcl['MedlineCitation']['Article']['Abstract']['AbstractText'][0])  #_abstracts = _res['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
#########################

#Preprocessing (depends on method)
ML_method = {
    "BoW": bag_of_words(_data, _pmidsList, _abstracts, False, (1,1)),
    #"KNN": knn('arg1', 'arg2')
}

#1 BAG OF WORDS
_preproc = ML_method["BoW"]
print(_preproc['class_results'])
#########################











