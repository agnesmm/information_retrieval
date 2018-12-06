from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel
from index import Index
from modeles import Weighter, Vectoriel, LanguageModel, Okapi
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer

import pprint
from tqdm import tqdm
import numpy as np
import operator


if __name__=='__main__':

    pp = pprint.PrettyPrinter()


    def eval(relevants, pred_docs):
        precisions = []

        for k, pred_doc in enumerate(pred_docs):
            if pred_doc in relevants:
                p = len(set(relevants).intersection(set(pred_docs[:k+1])))
                precisions.append(p/float(k+1))

        return np.mean(precisions)


    relevants = [1, 2, 3]
    pred_docs = [99, 1, 2]
    print(eval(relevants, pred_docs))




