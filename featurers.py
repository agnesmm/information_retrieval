import abc
import itertools

import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from index import Index
from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel
from modeles import PageRank, HITS, Okapi, IRmodel, Vectoriel
from modeles import LanguageModel
from weighter import SimpleWeighter, BasicWeighter, ThirdWeighter, FourthWeighter
from ParserCACM import ParserCACM, QueryParser
from utils import timeit

SEED = 0

def normalize(value, max_, min_):
    return (value-float(min_))/float(max_-min_)

class Featurer(object):

    __metaclass__ = abc.ABCMeta
    def __init__(self, index):
        self.index = index
        self.documents_count = float(index.documents_count)

    @abc.abstractmethod
    def get_features(self, idDoc, query):
        raise NotImplementedError

# model
class FeaturerOkapi(Featurer):
    def __init__(self, index):
        super(FeaturerOkapi, self).__init__(index)
        self.model = Okapi(self.index)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]


class FeaturerOkapi2(Featurer):
    def __init__(self, index):
        super(FeaturerOkapi2, self).__init__(index)
        self.model = Okapi(self.index, k1=1.8, b=.85)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]

class FeaturerLanguage(Featurer):
    def __init__(self, index):
        super(FeaturerLanguage, self).__init__(index)
        self.model = LanguageModel(self.index)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]


class FeaturerVectoriel1(Featurer):
    def __init__(self, index):
        super(FeaturerVectoriel1, self).__init__(index)
        self.model = Vectoriel(index, SimpleWeighter(index), normalized=True)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]

class FeaturerVectoriel2(Featurer):
    def __init__(self, index):
        super(FeaturerVectoriel2, self).__init__(index)
        self.model = Vectoriel(index, BasicWeighter(index), normalized=True)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]

class FeaturerVectoriel3(Featurer):
    def __init__(self, index):
        super(FeaturerVectoriel3, self).__init__(index)
        self.model = Vectoriel(index, ThirdWeighter(index), normalized=False)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]

class FeaturerVectoriel4(Featurer):
    def __init__(self, index):
        super(FeaturerVectoriel4, self).__init__(index)
        self.model = Vectoriel(index, FourthWeighter(index), normalized=False)
        self.features = {}

    def get_query_scores(self, query):
        ranks = self.model.getRanking(query.getText())
        return [(d, r/float(self.documents_count)) for (d,r) in ranks]

    def get_features(self, idDoc, query):
        queryId = query.getId()
        if queryId not in self.features.keys():
            self.features[queryId] = self.get_query_scores(query)
        return [s for (d, s) in self.features[queryId] if d==idDoc][0]



# document
class FeaturerDocLength(Featurer):
    def __init__(self, index):
        super(FeaturerDocLength, self).__init__(index)
        self.features = {}
        for idDoc in index.docs.keys():
            self.features[idDoc] = self.index.getDocsLength(idDoc)
        features_max = max(self.features.values())
        features_min = min(self.features.values())
        for idDoc in index.docs.keys():
            self.features[idDoc] = normalize(self.features[idDoc], features_max, features_min)

    def get_features(self, idDoc, query):
        return self.features[idDoc]

class FeaturerDocUniqueTermsCount(Featurer):
    def __init__(self, index):
        super(FeaturerDocUniqueTermsCount, self).__init__(index)
        self.features = {}
        for idDoc in index.docs.keys():
            self.features[idDoc] = len(self.index.getTfsForDoc(idDoc))
        features_max = max(self.features.values())
        features_min = min(self.features.values())
        for idDoc in index.docs.keys():
            self.features[idDoc] = normalize(self.features[idDoc], features_max, features_min)

    def get_features(self, idDoc, query):
        return self.features[idDoc]

class FeaturerDocPageRank(Featurer):
    def __init__(self, index):
        super(FeaturerDocPageRank, self).__init__(index)
        self.features = PageRank(index).getDocScores()
        for idDoc in index.docs.keys():
            self.features[idDoc] = self.features[idDoc]/float(self.documents_count)

    def get_features(self, idDoc, query):
        return self.features[idDoc]

class FeaturerDocHITS(Featurer):
    def __init__(self, index):
        super(FeaturerDocHITS, self).__init__(index)
        self.features = HITS(index).getDocScores()
        for idDoc in index.docs.keys():
            self.features[idDoc] = self.features[idDoc]/float(self.documents_count)

    def get_features(self, idDoc, query):
        return self.features[idDoc]


class FeaturerDocTermsIdf(Featurer):
    def __init__(self, index):
        super(FeaturerDocTermsIdf, self).__init__(index)
        self.features = {}
        #self.max_unique_term_count = 50.


    #@timeit
    def get_features(self, idDoc, query):
        #print('FeaturerDocTermsIdf')
        if idDoc not in self.features.keys():
            doc_terms = self.index.getTfsForDoc(idDoc).keys()
            self.features[idDoc] = sum([self.index.getIdfForStem(t) for t in doc_terms])
        return self.features[idDoc] / self.max_unique_term_count

#query
class FeaturerQueryUniqueTermsCount(Featurer):
    def __init__(self, index):
        super(FeaturerQueryUniqueTermsCount, self).__init__(index)
        self.features = {}
        self.max_unique_term_count = 5.

    #@timeit
    def get_features(self, idDoc, query):
        #print('FeaturerQueryUniqueTermsCount')
        if query not in self.features.keys():
            self.features[query] = len(self.index.textRepresenter.getTextRepresentation(query.getText()))
        return self.features[query] / self.max_unique_term_count

class FeaturerQueryIdf(Featurer):
    def __init__(self, index):
        super(FeaturerQueryIdf, self).__init__(index)
        self.features = {}
        self.max_idf_sum = 100.

    #@timeit
    def get_features(self, idDoc, query):
        #print('FeaturerQueryIdf')
        if query not in self.features.keys():
            query_terms = self.index.textRepresenter.getTextRepresentation(query.getText()).keys()
            self.features[query] = sum([self.index.getIdfForStem(t) for t in query_terms])
        return self.features[query] / self.max_idf_sum


class FeaturerQueryLength(Featurer):
    def __init__(self, index):
        super(FeaturerQueryLength, self).__init__(index)
        self.features = {}
        self.max_len_query = 30.

    #@timeit
    def get_features(self, idDoc, query):
        #print('FeaturerQueryLength')
        if query not in self.features.keys():
            query_terms = self.index.textRepresenter.getTextRepresentation(query.getText())
            self.features[query] = sum(query_terms.values())
        return self.features[query] / self.max_len_query



class FeaturersList(Featurer):
    def __init__(self, index):
        super(FeaturersList, self).__init__(index)
        self.featurers = [
                          FeaturerDocHITS(index),
                          FeaturerDocPageRank(index),
                          FeaturerOkapi(index),
                          FeaturerOkapi2(index),
                          FeaturerLanguage(index),
                          FeaturerVectoriel1(index),
                          FeaturerVectoriel2(index),
                          FeaturerVectoriel3(index),
                          FeaturerVectoriel4(index),
                          FeaturerDocLength(index),
                          FeaturerDocUniqueTermsCount(index),
                          # FeaturerDocTermsIdf(index),
                          # FeaturerQueryUniqueTermsCount(index),
                          # FeaturerQueryIdf(index),
                          # FeaturerQueryLength(index)
                          ]
        # self.featurers = [
        #           FeaturerOkapi(index),
        #           FeaturerOkapi2(index),
        #           FeaturerLanguage(index),
        #           FeaturerVectoriel1(index),
        #           FeaturerVectoriel2(index),
        #           # FeaturerDocLength(index),
        #           # FeaturerDocUniqueTermsCount(index),
        #           ]
    #@timeit
    def get_features(self, idDoc, query):
        # scores = []
        # for f in self.featurers:
        #     score = f.get_features(idDoc, query)
        #     scores.append(score)
        # return scores
        return [f.get_features(idDoc, query) for f in self.featurers]


class MetaModel(IRmodel):
    def __init__(self, index, featurers_list):
        super(MetaModel, self).__init__(index)
        self.featurers_list = featurers_list


class LinearMetaModel(MetaModel):
    def __init__(self, index, featurers_list):
        super(LinearMetaModel, self).__init__(index, featurers_list)

    def get_pred(self, docId, query, theta):
        features = np.array(self.featurers_list.get_features(docId, query))
        return np.dot(theta, features)

    def train_linear_model(self, alpha=10e-1,
                           lambda_=10e-5, t_max=1000,
                           decay=10e-7):



        rel_filename = 'cacm/cacm.rel'
        query_filename = 'cacm/cacm.qry'
        query_parser = QueryParser(rel_filename)
        query_parser.q.initFile(query_filename)
        train_queries, _ = query_parser.split_query_dataset()
        iterator = itertools.chain.from_iterable(itertools.repeat(train_queries,
                                                                  t_max/len(train_queries)))

        index = self.index
        featurers_count = len(self.featurers_list.featurers)
        docs_ids = index.docs.keys()
        theta = np.ones(featurers_count)
        N = len(docs_ids)

        base_model = LanguageModel(index)

        def get_f(theta, features):
            return np.dot(theta, features)

        random.seed(SEED)

        main_losses = []
        losses = []

        for i, query in enumerate(iterator):

            alpha *= (1. / (1. + decay * i))
            relevants = [d[1] for d in query.relevants]


            non_pertinent_doc = str(random.choice(docs_ids))
            while non_pertinent_doc in relevants:
                non_pertinent_doc = str(random.choice(docs_ids))

            non_pertinent_docs = [non_pertinent_doc]

            # base_model_preds = [d for d,rank in base_model.getRanking(query.getText())[:20]]
            # non_pertinent_docs = [str(d) for d in base_model_preds if int(d) not in relevants]
            # non_pertinent_docs = random.sample(non_pertinent_docs, 1)

            for non_pertinent_doc in non_pertinent_docs:
                pertinent_doc = str(random.choice(relevants))
                pertinent_doc_features = np.array(self.featurers_list.get_features(pertinent_doc, query))
                non_pertinent_doc_features = np.array(self.featurers_list.get_features(non_pertinent_doc, query))
                f_pertinent_doc = get_f(theta, pertinent_doc_features)
                f_non_pertinent_doc = get_f(theta, non_pertinent_doc_features)
                loss = 1 - f_pertinent_doc + f_non_pertinent_doc
                losses.append(loss>0)
                if loss > 0:
                    theta += alpha*(pertinent_doc_features-non_pertinent_doc_features)
                    theta *= (1-lambda_*np.linalg.norm(theta, 2))


            if i%100==0:
                print(i)
                print('regul', (1-lambda_*np.linalg.norm(theta, 2)))
                print('lr', alpha)
                print(theta)
                t_loss = np.mean(losses)
                print('LOSS = ', t_loss)
                main_losses.append(t_loss)
                losses = []
        print(main_losses)
        plt.plot(list(range(len(main_losses))), main_losses)
        plt.title('Average loss')
        plt.savefig('plot/meta_model_loss_alpha_=0_5_decay_8')
        return theta


    def getScores(self, query, theta):
        scores = {}
        # docs = [d for d,rank in Okapi(self.index).getRanking(query.getText())][:500]
        docs = index.docs.keys()
        for docId in docs:
            scores[docId] = self.get_pred(docId, query, theta)
        return scores




if __name__=='__main__':



    index = Index()


    featurers_list = FeaturersList(index)

    model = LinearMetaModel(index, featurers_list)
    theta = model.train_linear_model(alpha=.02, t_max=800,
                                     lambda_=.001, decay=10e-8)
    # theta = [2.46331609, 8.24653025, 6.11832922, 0.59504227, -2.89611445, 4.59988454, 1.91155859, 1.62453584]

    random.seed(0)

    rel_filename = 'cacm/cacm.rel'
    query_filename = 'cacm/cacm.qry'
    query_parser = QueryParser(rel_filename)
    query_parser.q.initFile(query_filename)
    _, test_queries = query_parser.split_query_dataset()


    print('EVAL')
    queries = []
    for query in test_queries:
        scores = model.getRanking(query, theta)
        queries.append(IRList(query, scores))
    scores = EvalIRModel().eval_model(queries)
    print(scores)
    plt.figure(figsize=(10,8))


    y = scores['PrecisionRecall']['mean']
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.title('Interpolated precision on metamodel')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig('plot/interpolated_p_metamodel.png')


