import xlrd
import numpy as np 
import pandas as pd
import requests
from Bio import Entrez
from Method_Handler import MethodHandler

def importFrom(type='xl', *args):
    return importHandler.execMethod(type, None, *args)

def xl_import(file_path):
    wb = xlrd.open_workbook(file_path)
    sheet = wb.sheet_by_index(0)
    d = {}
    for col in range(sheet.ncols):
        _col = []
        for row in range(sheet.nrows):
            if row != 0:
                _col.append(sheet.cell_value(row, col))
        d[sheet.cell_value(0, col)] = _col    
    return pd.DataFrame.from_dict(d)

def csv_import(file_path):
    d = {}
    return pd.DataFrame.from_dict(d)

def pkl_import(file_path):    
    return pd.read_pickle(file_path)

def pubmedInfoDownload(data):    
    PMIDS = ','.join(data['pmid'].to_list())
    Entrez.email = 'marcarmenter@hotmail.com'
    query = Entrez.efetch(db='pubmed', retmode='xml', report='abstract', id=PMIDS) #PMIDS, 24192311,23534783
    res = readEntrezQuery(Entrez.read(query))
    res['Candidate'] = data['n'] #Add ico info from data
    return res

def readEntrezQuery(EntrezQuery):
    for i,a in enumerate(EntrezQuery['PubmedArticle']):
        citation = a['MedlineCitation']
        art_data = dict({
            'PMID': [str(citation['PMID'])],
            'DateCompleted': [citation['DateCompleted']['Year'] + '/'
                            + citation['DateCompleted']['Month'] + '/'
                            + citation['DateCompleted']['Day']] if 'DateCompleted' in citation else [[]],
            'DateRevised': [citation['DateRevised']['Year'] + '/'
                            + citation['DateRevised']['Month'] + '/'
                            + citation['DateRevised']['Day']] if 'DateRevised' in citation else [[]],
            'JournalTitle': [str(citation['Article']['Journal']['Title'])],
            'JournalCountry': [str(citation['MedlineJournalInfo']['Country'])],
            'KeywordList': [citation['KeywordList']],
            'MeshHeadingList': [', '.join(d['DescriptorName'] for d in citation['MeshHeadingList'])] if 'MeshHeadingList' in citation else [[]],
            'Lang': [', '.join(citation['Article']['Language'])],
            'PublicationDate': [citation['Article']['Journal']['JournalIssue']['PubDate']['Year']
                            + (' ' + citation['Article']['Journal']['JournalIssue']['PubDate']['Month']) if 'Month' in citation['Article']['Journal']['JournalIssue']['PubDate'] else ''],
            'ArticleTitle': [str(citation['Article']['ArticleTitle'])],
            'Abstract': [' '.join(citation['Article']['Abstract']['AbstractText'])],
            'Authors': [', '.join((auth['ForeName'] + ' ' + auth['LastName']) if 'ForeName' in auth and 'LastName' in auth else auth['CollectiveName'] if 'CollectiveName' in auth else auth['LastName'] for auth in citation['Article']['AuthorList'])]
        })
        if 'df' not in locals(): 
            df = pd.DataFrame.from_dict(art_data)
        else: 
            df = df.append(pd.DataFrame.from_dict(art_data), ignore_index=True)
    return df

def saveFile(data, path):
    try:
        data.to_pickle(path)
    except:
        print('Unable to save file')

#isNAN
def isNaN(val):
    val = val.replace(",", "")
    try:
        val = float(val)
    except:
        pass
        return True
    return val != val
####################

#getNContext
def getNContext(atcl, index, n):
    if index < n:
        context = atcl[:index]
    else:
        context = atcl[(index-n):index]
    if n > len(atcl[index:]):
        context.extend(atcl[index:])
    else:
        context.extend(atcl[index:(index+n)])
    return context
####################

#pad_array
def pad_array(array, length):
    if array == None: array = []
    if array.__len__() < length:
       array.extend([0] * (length - array.__len__()))
    return array

def candidateClsf(preDF, candDF):
    if not isinstance(preDF, pd.DataFrame) or not isinstance(candDF, pd.DataFrame):
        raise Exception('Bad data type - \'dataframe\' expected')
    preDF['Is Candidate'] = getCandidatesDF(preDF, candDF)
    return preDF

#this function is build taking into account that both lists have same PMID order
def getCandidatesDF(preDF, candDF):
    candList = []
    for i, n in enumerate(preDF['N'].to_list()):
        found = False
        prePmid = preDF['PMID'][i]
        if i == 0 or prePmid != candPmid: 
            if i != 0 and prePmid != candPmid:
                candDF = candDF.drop(candDF.index[0])
            candPmid = candDF['PMID'].iloc[0]                       
            cand = candDF['Candidate'].to_list()[0] #take candidate
            cand = [str(int(cand))] if isinstance(cand, float) else  cand.split(',') #convert to list of candidates            
        if candPmid == prePmid and n in cand:
            found = True
        candList.append(1 if found else 0)
    return candList

importMethodDicc = {
    'xl': xl_import,
    'csv': csv_import,
    'pkl': pkl_import
}

importHandler = MethodHandler(importMethodDicc)

