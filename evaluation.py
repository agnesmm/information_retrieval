import abc
import matplotlib.pyplot as plt
import numpy as np

from utils import timeit

class IRList(object):
    def __init__(self, query, doc_scores):
        self.query = query
        self.doc_scores = doc_scores

class EvalMeasure(object):

    @abc.abstractmethod
    def eval(l):
        raise NotImplementedError


class P(EvalMeasure):

    # @timeit
    def eval(self, l, nbLevels=10):

        relevants = [r['doc_id'] for r in l.query.relevants]
        relevants = list(set(relevants)) ## ok ?
        total = len(relevants)

        pred_docs = [int(d[0]) for d in l.doc_scores]
        precision, recall = [], []

        for k in range(1, len(pred_docs)+1):
            right_doc_count = len(set(relevants).intersection(pred_docs[:k]))
            precision.append(right_doc_count/float(k))
            recall.append(right_doc_count/float(total))

        interpolated_recall = np.linspace(0, 1, nbLevels+1)

        precision = np.array(precision)
        interpolated_precision = []
        for i_recall in interpolated_recall:
            max_prec = max(precision[recall>=i_recall])
            interpolated_precision.append(max_prec)
        return np.array(interpolated_precision)


class AP(EvalMeasure):
    def eval(self, l):

        relevants = [r['doc_id'] for r in l.query.relevants]
        pred_docs = [int(d[0]) for d in l.doc_scores]
        precisions = []

        for k, pred_doc in enumerate(pred_docs):
            if pred_doc in relevants:
                p = len(set(relevants).intersection(set(pred_docs[:k+1])))
                precisions.append(p/float(k+1))
        return np.mean(precisions)


class P20(EvalMeasure):
    def eval(self, l):
        # relevants = [r[1] for r in l.query.relevants]
        relevants = [r['doc_id'] for r in l.query.relevants]
        pred_docs = [int(d[0]) for d in l.doc_scores][:20]
        p20 = len(set(relevants).intersection(set(pred_docs)))/20.0
        return p20


class CR20(EvalMeasure):
    def eval(self, l):
        relevant_subtopic = list(set([r['idSubtopic'] for r in l.query.relevants]))
        subtopic_count = len(relevant_subtopic)

        pred_docs = [int(d[0]) for d in l.doc_scores][:20]
        predicted_subtopic = list(set([r['idSubtopic'] for r in l.query.relevants if r['doc_id'] in pred_docs]))

        print('relevant_subtopic = %d // predicted_subtopic = %d'%(subtopic_count, len(predicted_subtopic)))

        cr20 = len(predicted_subtopic) / float(subtopic_count)
        return cr20

class EvalIRModel(object):
    def __init__(self, measures = ['PrecisionRecall', 'AveragePrecision']):

        measures_func = {'PrecisionRecall': P(),
                          'AveragePrecision': AP(),
                          'P@20': P20(),
                          'CR@20': CR20()}

        self.measures = {m: measures_func[m] for m in measures}

    def eval_model_with_measure(self, queries, measure):

        # for l in queries:
        #     print(measure.eval(l))
        p = np.array([measure.eval(l) for l in queries])
        return {'eval_std': np.std(p, axis=0),
                'mean': np.mean(p, axis=0)}

    def eval_model(self, queries):
        evaluation = {}
        for name, measure in self.measures.items():
            evaluation[name] = self.eval_model_with_measure(queries, measure)
        return evaluation


