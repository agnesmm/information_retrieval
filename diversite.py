import abc
from collections import Counter
from random import randint
from utils import roundrobin
import operator
import os

import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans

from index import Index
from modeles import IRmodel, Okapi
from ParserCACM import ParserCACM, QueryParser


class Clustering(object):
    def __init__(self, index):
        self.index = index

    def get_docs_tfs(self, docs_ids):
        tfs = [self.index.getTfsForDoc(docId) for docId in docs_ids]
        matrix_tfs = pd.DataFrame(tfs, index=docs_ids).fillna(0)
        return np.array(matrix_tfs)

    def cluster_documents(self, docs_ids):
        tfs = self.get_docs_tfs(docs_ids)
        clusters = self.clustering_algo.fit_predict(tfs)
        return {docId: cluster for docId, cluster in zip(docs_ids, clusters)}

class RandomClustering(Clustering):
    def __init__(self, index):
        super(RandomClustering, self).__init__(index)

    def cluster_documents(self, docs_ids, n_clusters=20):
        return {docId: randint(0, n_clusters) for docId in docs_ids}

class KmeanClustering(Clustering):
    def __init__(self, index, n_clusters=10):
        super(KmeanClustering, self).__init__(index)
        self.clustering_algo = KMeans(n_clusters=n_clusters)

class DBSCANClustering(Clustering):
    def __init__(self, index, eps=.3, min_samples=3):
        super(DBSCANClustering, self).__init__(index)
        self.clustering_algo = DBSCAN(eps=eps, min_samples=min_samples)

class WithoutClustering(Clustering):
    def __init__(self, index, eps=.3, min_samples=3):
        super(WithoutClustering, self).__init__(index)
        self.clustering_algo = DBSCAN(eps=eps, min_samples=min_samples)

    def cluster_documents(self, docs_ids):
        return {doc_id: i for i, doc_id in enumerate(docs_ids)}

class DiversityClustering(IRmodel):
    def __init__(self, clustering, index, N=100):
        super(DiversityClustering, self).__init__(index)
        self.clustering = clustering(index)
        self.N = N #nbr of docs to cluster
        # quel clustering choisir ? pourquoi ? tester pls clustering
        # par exemple prendre un intervalle et faire un pas pour le k
        # comparer des algo, lequels sont adaptes ?
        pass


    def similarity(self, docId, pivot):
        pass

    def order_cluster_by_docs_count(self, clusters, docs_scores):
        docs_clusters = [clusters[doc_id] for (doc_id, _) in docs_scores[:self.N]]
        ordered_clusters = sorted(Counter(docs_clusters).items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        return [cluster_id for cluster_id, cluster_freq in ordered_clusters]


    def order_cluster_by_relevance(self, clusters, docs_scores):
        cluster_relevance = {}
        for cluster_id in set(list(clusters.values())):
            docs_in_cluster = [doc_id for doc_id, c in clusters.items() if c==cluster_id]
            docs_relevance = [score for doc_id, score in docs_scores if doc_id in docs_in_cluster]
            cluster_relevance[cluster_id] = min(docs_relevance)
        ordered_clusters = sorted(Counter(cluster_relevance).items(),
                          key=operator.itemgetter(1),
                          reverse=False)
        return [cluster_id for cluster_id, cluster_freq in ordered_clusters]


    def order_docs_by_relevance(self, docs_ids, docs_scores):
        docs_relevance = {doc_id:score for doc_id, score in docs_scores if doc_id in docs_ids}
        ordered_docs = sorted(Counter(docs_relevance).items(),
                  key=operator.itemgetter(1),
                  reverse=False)
        return [doc_id for doc_id, doc_freq in ordered_docs]

    def order_pred(self, query, docs_scores, cluster_order='docs_count'):
        docs_to_cluster = [doc_id for (doc_id, _) in docs_scores[:self.N]]
        clusters = self.clustering.cluster_documents(docs_to_cluster)
        docs_ranked = []

        if cluster_order=='docs_count':
            ordered_clusters = self.order_cluster_by_docs_count(clusters, docs_scores)

        elif cluster_order=='relevance':
            ordered_clusters = self.order_cluster_by_relevance(clusters, docs_scores)

        else:
            print('implement cluster order!!')
            exit()

        docs_grouped_by_clusters = []
        for cluster_id in ordered_clusters:
            clusters_docs = [d for d,c in clusters.items() if c==cluster_id]
            print(clusters_docs)
            clusters_docs_ordered = self.order_docs_by_relevance(clusters_docs, docs_scores)
            print(clusters_docs_ordered)
            exit()
            docs_grouped_by_clusters.append(clusters_docs_ordered)

        ordered_docs = list(roundrobin(*docs_grouped_by_clusters))
        return [(doc_id, rank) for rank, doc_id in enumerate(ordered_docs)]



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

    # def order_pred(self, query, docs_scores):
        # docsIds = [docId for docId,_ in docs_scores]
        # ordered_doc = []

        # for docId, k in docs_scores:
        #     print('docId', docId, 'k', k)
        #     if k == 0:
        #         d.append(docId)
        #         pass

        #     pivot = docsIds[:k]

        #     d_i = similarity(docId, query, pivot)
        #     ordered_doc.append(d_i)
        #     pivot = [k for k in pivot if k != d_i]

        #     print(pivot)
        #     print(k, docId)
        #     exit()
        # return

def main():
    pp = pprint.PrettyPrinter()

    data_dir = 'data/easyCLEF08'
    txt_file = os.path.join(data_dir, 'easyCLEF08_text.txt')
    query_file = os.path.join(data_dir, 'easyCLEF08_query.txt')
    relevants_file = os.path.join(data_dir, 'easyCLEF08_gt.txt')

    index = Index(directory=data_dir,
                  txt_file=txt_file,
                  create_index=False)

    okapi = Okapi(index, k1=1.80, b=0.65)

    diversity = DiversityClustering(DBSCANClustering, index)

    q = QueryParser(relevants_file)
    q.q.initFile(query_file)

    # while True:
    for i in range(1):
        query = q.nextQuery()
        if query is None:
            break
        if len(query.relevants)==0:
            print('No relevants docs')
            continue

        docs_scores = okapi.getRanking(query.getText())
        ordered_pred = diversity.order_pred(query, docs_scores)


if __name__ == '__main__':
    main()







