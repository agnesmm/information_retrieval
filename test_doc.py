from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel
from index import Index
from modeles import Weighter, Vectoriel, Okapi
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer


if __name__=='__main__':

    rel_filename = 'cacm/cacm.rel'
    query_filename = 'cacm/cacm.qry'

    index = Index(name='test', docFrom=None,
                  parser=ParserCACM, textRepresenter=PorterStemmer,
                  create_index=False)


    weighter = Weighter(index)
    parser = ParserCACM()
    parser.initFile('cacm/cacm.txt')
    doc = parser.nextDocument()
    print(doc.others['links'])
    # for d in range(20,22):
    #     docId = str(d)
    #     print(ParserCACM().getDocument(docId))
        # print(weighter.getDocWeightsForDoc(docId), index.getDocsLength(docId))

    # q = QueryParser(query_filename, rel_filename)
    # train_queries, test_queries = q.split_query_dataset()
    # print(len(train_queries), len(test_queries))









