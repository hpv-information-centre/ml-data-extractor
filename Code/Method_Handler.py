
class MethodHandler:

    def __init__(self, methodDicc=None, dictNames=[]):
        self.methodDicc = {}
        if isinstance(methodDicc, list):
            if not isinstance(dictNames, list) or dictNames.__len__() < methodDicc.__len__():
                defNames = [str(x + 1) for x in range(methodDicc.__len__())]
                if dictNames.__len__() > 0:
                    for k, n in enumerate(dictNames):
                        defNames[k] = n
                dictNames = defNames
            for i, d in enumerate(methodDicc):
                if not isinstance(d, dict):
                    print('\nSome element of the list is not a dict, it will be set empty.\n')
                    d = {}
                self.methodDicc[str(dictNames[i])] = d
        elif isinstance(methodDicc, dict):
            self.methodDicc = methodDicc
        else:
            print('\nThe parameter methodDiccList is not a list nor dict, dict will be set empty.\n')

    # def getMethodNames(self):
    #     return list(self.methodDicc.keys())
    
    def execMethod(self, method, methodDicc=None, *args):
        try:
            if methodDicc == None:
                item = self.methodDicc[method]
            else:
                item = self.methodDicc[methodDicc][method]
        except KeyError:
                return None
        return item(*args)