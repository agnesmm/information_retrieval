import abc
import math

class Weighter(object):
    def __init__(self, index):
        self.index = index
        self.textRepresenter = self.index.textRepresenter

    @abc.abstractmethod
    def getDocWeightsForStem(self, stem):
        raise NotImplementedError

    @abc.abstractmethod
    def getDocWeightsForDoc(self, idDoc):
        raise NotImplementedError

    @abc.abstractmethod
    def getWeightsForQuery(self, query):
        raise NotImplementedError



# premier de la liste
class BasicWeighter(Weighter):
    def __init__(self, index):
        super(BasicWeighter, self).__init__(index)

    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)

    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)

    def getWeightsForQuery(self, query):
        query_weights = self.textRepresenter.getTextRepresentation(query)
        return {t:1 for t,_ in query_weights.items()}

# deuxieme de la liste
class SimpleWeighter(Weighter):
    def __init__(self, index):
        super(SimpleWeighter, self).__init__(index)

    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)

    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)

    def getWeightsForQuery(self, query):
        return self.textRepresenter.getTextRepresentation(query)


# troisieme de la liste
class ThirdWeighter(Weighter):
    def __init__(self, index):
        super(ThirdWeighter, self).__init__(index)

    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)

    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)

    def getWeightsForQuery(self, query):
        query_weights = self.textRepresenter.getTextRepresentation(query)
        return {t: self.index.getIdfForStem(t) for t in query_weights.keys()}


# quatrieme de la liste
class FourthWeighter(Weighter):
    def __init__(self, index):
        super(FourthWeighter, self).__init__(index)

    def getDocWeightsForStem(self, stem):
        stem_tfs = self.index.getTfsForStem(stem)
        return {d:1+math.log(s) for d,s in stem_tfs.items()}

    def getDocWeightsForDoc(self, idDoc):
        doc_weights = self.index.getTfsForDoc(idDoc)
        return {t: 1+math.log(w) for t,w in doc_weights.items()}

    def getWeightsForQuery(self, query):
        query_weights = self.textRepresenter.getTextRepresentation(query)
        return {t: self.index.getIdfForStem(t) for t in query_weights.keys()}


