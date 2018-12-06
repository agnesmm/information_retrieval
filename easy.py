from diversite import DiversityClustering, RandomClustering
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

    models = [Okapi(index, k1=1.80, b=0.65)]
    models_names = ['Okapi']

    # models = [Vectoriel(index, BasicWeighter(index), normalized=True),
    #           Okapi(index),
    #           LanguageModel(index, .7),
    #           HITS(index, seeds=3, k=5),
    #           PageRank(index, seeds=3, k=5)
    #           ]

    # models_names = ['Vectoriel', 'Okapi', 'LanguageModel', 'HITS', 'PageRank']


    diversity = DiversityClustering(RandomClustering, index)

    models_scores = {}

    for model, name in zip(models, models_names):
        print('Evaluation of %s model'%(name))

        q = QueryParser(relevants_file)
        q.q.initFile(query_file)
        predicted_docs = []

        # while True:
        for i in range(3):
            query = q.nextQuery()
            if query is None:
                break
            if len(query.relevants)==0:
                print('No relevants docs')
                continue


            docs_scores = model.getRanking(query.getText())
            query_pred = IRList(query, docs_scores)


            ordered_pred = diversity.order_pred(query, docs_scores)
            ordered_query_pred = IRList(query, ordered_pred)

            # predicted_docs.append(query_pred)
            predicted_docs.append(ordered_query_pred)

        models_scores[name] = EvalIRModel(measures=['CR@20', 'P@20', 'AveragePrecision']).eval_model(predicted_docs)
        # models_scores[name] = EvalIRModel().eval_model(predicted_docs)

    pp.pprint(models_scores)




