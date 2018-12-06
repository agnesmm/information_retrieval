import abc
from collections import Counter
from random import randint
from utils import roundrobin
import operator

from modeles import IRmodel

class Clustering(object):
    def __init__(self, index):
        self.index = index

    @abc.abstractmethod
    def cluster_documents(self):
        raise NotImplementedError

class KmeanClustering(Clustering):
    def __init__(self, index):
        super(KmeanClustering, self).__init__(index)

    def cluster_documents(self):
        # todo
        pass


class RandomClustering(Clustering):
    def __init__(self, index):
        super(RandomClustering, self).__init__(index)

    def cluster_documents(self, k=100):
        docs_ids = self.index.docs.keys()
        cluster = {docId: randint(0, k) for docId in docs_ids}
        return cluster




class DiversityClustering(IRmodel):
    def __init__(self, clustering, index):
        super(DiversityClustering, self).__init__(index)
        c = clustering(index)
        self.clusters = c.cluster_documents()
        # quel clustering choisir ? pourquoi ? tester pls clustering
        # par exemple prendre un intervalle et faire un pas pour le k
        # comparer des algo, lequels sont adaptes ?
        pass


    def similarity(self, docId, pivot):
        pass

    def order_pred(self, query, docs_scores, cluster_order='docs_count'):
        # def un ordre des docs par cluster

        clusters = self.clusters
        docs_ranked = []

        docsIds = [docId for docId,_ in docs_scores]
        docs_clusters = [clusters[docId] for docId in docsIds]

        if cluster_order=='docs_count':
            ordered_clusters = sorted(Counter(docs_clusters).items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
            ordered_clusters = [cluster_id for cluster_id, cluster_freq in ordered_clusters]

        else:
            print('implement cluster order!!')
            exit()

        docs_grouped_by_clusters = []
        for cluster_id in ordered_clusters:
            docs_grouped_by_clusters.append([d for d,c in clusters.items() if c==cluster_id])

        return list(roundrobin(*docs_grouped_by_clusters))



class Glouton(IRmodel):
    def __init__(self, clustering, index):
        c = clustering(index)
        self.clusters = c.cluster_documents()
        # quel clustering choisir ? pourquoi ? tester pls clustering
        # par exemple prendre un intervalle et faire un pas pour le k
        pass


    def value(self, docId, query, pivot):
        # how to define value fonction
        # implem differente value
        pass

    def order_pred(self, query, docs_scores):
        docsIds = [docId for docId,_ in docs_scores]
        ordered_doc = []

        for docId, k in docs_scores:
            print('docId', docId, 'k', k)
            if k == 0:
                d.append(docId)
                pass

            pivot = docsIds[:k]

            d_i = similarity(docId, query, pivot)
            ordered_doc.append(d_i)
            pivot = [k for k in pivot if k != d_i]

            print(pivot)
            print(k, docId)
            exit()
        return




