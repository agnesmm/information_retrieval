from diversite import DiversityClustering, RandomClustering, DBSCANClustering
from diversite import KmeanClustering, WithoutClustering
from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel, P20
from index import Index
from modeles import Vectoriel, LanguageModel, Okapi, HITS, PageRank
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer
from weighter import Weighter, BasicWeighter

import pprint
from tqdm import tqdm
import numpy as np
import operator
import os

if __name__=='__main__':
    pp = pprint.PrettyPrinter()

    data_dir = 'data/easyCLEF08'
    txt_file = os.path.join(data_dir, 'easyCLEF08_text.txt')
    query_file = os.path.join(data_dir, 'easyCLEF08_query.txt')
    relevants_file = os.path.join(data_dir, 'easyCLEF08_gt.txt')

    index = Index(directory=data_dir,
                  txt_file=txt_file,
                  create_index=False)

    #diversity_models = [DiversityClustering(WithoutClustering, index, N=100),
    #                    DiversityClustering(RandomClustering, index, N=100),
    #                    DiversityClustering(KmeanClustering, index, N=100),
    #                    DiversityClustering(DBSCANClustering, index, N=100)]
    #names = ['Without', 'Random', 'Kmeans', 'DBSCAN']

    diversity_models = [DiversityClustering(KmeanClustering, index, N=100)]
    names = ['Kmeans']

    models_scores = {}

    for diversity, name in zip(diversity_models, names):
        print('Evaluation of %s clustering'%(name))

        q = QueryParser(relevants_file)
        q.q.initFile(query_file)
        predicted_docs = []

        while True:
        # for i in range(3):
            query = q.nextQuery()

            if query is None:
                break
            if len(query.relevants)==0:
                print('No relevants docs')
                continue

            docs_scores = Okapi(index, k1=1.80, b=0.65).getRanking(query.getText())
            ordered_pred = diversity.order_pred(query, docs_scores, cluster_order='relevance')
            predicted_docs.append(IRList(query, ordered_pred))

        models_scores[name] = EvalIRModel(measures=['CR@20', 'P@20', 'AveragePrecision']).eval_model(predicted_docs)

    pp.pprint(models_scores)

