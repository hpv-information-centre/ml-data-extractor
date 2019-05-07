import xlrd
import numpy as np 
import pandas as pd
import re

# def sort_coo(m):
#     tuples = zip(m.row, m.col, m.data)
#     return sorted(tuples, key=lambda x: (x[0], x[2]))

#READ EXCEL - Returns numpy matrix
def xl_export(file_path):
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
####################

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
    atcl = [re.sub('[(),.:;!?[\]]', '', s) for s in atcl]
    atcl = list(filter(None, atcl))

    if index < n:
        context = atcl[:index]
    else:
        context = atcl[(index-n):index]
    if n > len(atcl[index:]):
        context.extend(atcl[index:])
    else:
        context.extend(atcl[index:(index+n)])
    return ' '.join(str(x) for x in context)
####################

#pad_array
def pad_array(array, length):
    if array == None:
        array = []

    if len(array) < length:
        return array + [0] * (length - len(array))
    else:
        return array
            