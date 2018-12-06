from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel
from index import Index
from modeles import Vectoriel, Okapi, LanguageModel
from modeles import PageRank, HITS
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer
from weighter import SimpleWeighter, BasicWeighter, ThirdWeighter, FourthWeighter

import numpy as np

if __name__=='__main__':


    rel_filename = 'data/cacm/cacm.rel'
    query_filename = 'data/cacm/cacm.qry'

    index = Index()




    # models = [
    #   Vectoriel(index, BasicWeighter(index), normalized=True),
    #   Vectoriel(index, SimpleWeighter(index), normalized=True),
    #   Vectoriel(index, ThirdWeighter(index), normalized=True),
    #   Vectoriel(index, FourthWeighter(index), normalized=True),
    #   ]
    # models = [Okapi(index, k1=1.80, b=0.65), LanguageModel(index, lissage=.8)]

    # model = LanguageModel(index)

    # base_model = Okapi(index)

    # model1 = PageRank(index, seeds=5, k=2)


    models = [HITS(index, seeds=5, k=2), HITS(index, seeds=5, k=5)]

    for model in models:
        q = QueryParser(rel_filename)
        q.q.initFile(query_filename)
        queries = []
        while True:
        # for i in range(4):
            query = q.nextQuery()
            if query is None:
                break
            if len(query.relevants)==0:
                print('No relevants docs')
                continue


            scores = model.getRanking(query.getText())


            queries.append(IRList(query, scores))
        print('___________')

    print(model)
    print(EvalIRModel().eval_model(queries))
    print('\n')
    print('\n')








