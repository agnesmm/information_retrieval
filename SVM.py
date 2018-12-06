import random

import numpy as np
from tqdm import tqdm

from index import Index
from ParserCACM import ParserCACM, QueryParser
from featurers import MetaModel, FeaturersList

SEED = 0

class MetaModelSVM(MetaModel):
    def __init__(self, index, featurers_list):
        super(MetaModelSVM, self).__init__(index, featurers_list)


    def write_data(self):
        filename = 'data/train.dat'
        with open(filename, 'w') as file:
            query_parser = QueryParser('cacm/cacm.rel')
            query_parser.q.initFile('cacm/cacm.qry')
            train_queries, _ = query_parser.split_query_dataset()

            index = self.index
            featurers_count = len(self.featurers_list.featurers)
            docs_ids = index.docs.keys()
            N = len(docs_ids)

            random.seed(SEED)

            for _, query in enumerate(train_queries):
                queryId = query.getId()
                relevants = [str(d[1]) for d in query.relevants]
                n_relevants = len(relevants)

                for i, pertinent_doc in enumerate(relevants):
                    non_pertinent_doc = str(random.choice(docs_ids))
                    while non_pertinent_doc in relevants:
                        non_pertinent_doc = str(random.choice(docs_ids))

                    pertinent_doc_features = self.featurers_list.get_features(pertinent_doc, query)
                    non_pertinent_doc_features = self.featurers_list.get_features(non_pertinent_doc, query)
                    line1 = '{} qid:{}'.format(1, queryId)
                    for f_idx,f in enumerate(pertinent_doc_features):
                        line1 += ' %d:%.7f'%(f_idx,f)
                    line1 += '\n'
                    file.write(line1)
                    rank = i+2
                    line2 = '{} qid:{}'.format(rank, queryId)
                    for f_idx,f in enumerate(non_pertinent_doc_features):
                        line2 += ' %d:%.7f'%(f_idx,f)
                    line2 += '\n'
                    file.write(line2)

if __name__=='__main__':
    index=Index()
    featurers_list = FeaturersList(index)
    m = MetaModelSVM(index, featurers_list)
    m.write_data()



