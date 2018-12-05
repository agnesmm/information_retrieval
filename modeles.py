#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Document import Document
from index import Index
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer
from utils import timeit

import abc
import ast
import json
import math
import numpy as np
import operator
from random import randint
import random


class IRmodel(object):
    def __init__(self, index):
        self.index = index

    @abc.abstractmethod
    def getScores(self, query):
        raise NotImplementedError

    def getRanking(self, query, *args):
        scores =  self.getScores(query, *args)
        # add documents not in scores
        docsIds = self.index.docs.keys()
        default_score = min(scores.values()) - 1
        all_scores = {d: scores.get(d, default_score) for d in docsIds}
        ordered_scores = self.order_dict(all_scores)
        return [(docId, rank) for rank, (docId, _) in enumerate(ordered_scores)]

    def order_dict(self, dico):
        return sorted(dico.items(),
          key=operator.itemgetter(1),
          reverse=True)


class Vectoriel(IRmodel):
    def __init__(self, index, weighter, normalized=False):
        super(Vectoriel, self).__init__(index)
        self.weighter = weighter
        self.normalized = normalized

    def normalize_dict(self, dico):
        norm = float(sum(dico.values()))
        return {k:v/norm for k, v in dico.items()}

    def getScores(self, query, *args):
        normalized=self.normalized
        scores = {}
        query_weights = self.weighter.getWeightsForQuery(query)
        if normalized:
            query_weights = self.normalize_dict(query_weights)

        for t, t_weight in query_weights.items():
            docs_weights = self.weighter.getDocWeightsForStem(t)
            if normalized:
                docs_weights = self.normalize_dict(docs_weights)

            for doc, doc_weight in docs_weights.items():
                scores[doc] = scores.get(doc,0.0)+(t_weight*doc_weight)

        return scores


class LanguageModel(IRmodel):
    def __init__(self, index, lissage=.7):
        super(LanguageModel, self).__init__(index)
        self.lissage=lissage

    def getScores(self, query):
      lissage = self.lissage
      index = self.index
      corpus_length = self.index.corpus_length

      scores = {}
      cumulated_penalty = 0
      query_weights = index.textRepresenter.getTextRepresentation(query)

      for t, tf in query_weights.items():
          t_docs = index.getTfsForStem(t)
          p_corpus = sum(t_docs.values())/corpus_length
          if p_corpus == 0:
              continue
          else:
              t_penalty = tf*math.log(lissage*p_corpus)

          for docId, tf_doc in t_docs.items():
              p_doc = tf_doc/float(index.getDocsLength(docId))
              f_t = tf*math.log((1.-lissage)*p_doc + lissage*p_corpus)
              scores[docId] = scores.get(docId, cumulated_penalty) + f_t

          cumulated_penalty += t_penalty
          missing_docs = [d for d in scores.keys() if d not in t_docs.keys()]

          for docId in missing_docs:
                  scores[docId] = scores.get(docId) + t_penalty
      return scores

class Okapi(IRmodel):
    def __init__(self, index, k1=1.8, b=.85):
        super(Okapi, self).__init__(index)
        self.Lmoy = self.get_average_docs_length()
        self.N = len(self.index.docs.keys())
        self.k1 = k1
        self.b = b

    def get_average_docs_length(self):
        docLengths = [self.index.getDocsLength(docId) for docId in self.index.docs]
        return float(np.mean(docLengths))

    def getScores(self, query):
        scores = {}
        Lmoy = self.Lmoy
        k1 = self.k1
        b = self.b
        N = self.N
        query_weights = self.index.textRepresenter.getTextRepresentation(query)

        for t, _ in query_weights.items():
            docs_weights = self.index.getTfsForStem(t)
            n = len(docs_weights.keys())
            idf = max(0., math.log((N-n+.5)/(n+.5)))
            for docId, doc_weight in docs_weights.items():
                L_doc = self.index.getDocsLength(docId)
                numerateur = (k1+1.)*doc_weight
                denom = k1*(1. - b + b * L_doc/Lmoy) + doc_weight
                scores[docId] = scores.get(docId,0.0) + idf*(numerateur / denom)

        return scores



class RandomWalk(IRmodel):
    def __init__(self, index, base_model, seeds=20, k=5):
        super(RandomWalk, self).__init__(index)
        self.base_model = base_model
        self.seeds = seeds
        self.k = k
        self.docs_links = self.docs_links_matrix()


    def docs_links_matrix(self):
        docs = self.index.docs
        doc_ids = list(self.index.docs.keys())
        doc_id2idx = {docId:idx for idx, docId in enumerate(doc_ids)}
        doc_idx2id = {idx:docId for docId, idx in doc_id2idx.items()}
        self.doc_id2idx = doc_id2idx
        self.doc_idx2id = doc_idx2id

        N = len(docs)
        M = np.zeros((N,N))
        for docId in doc_ids:
            docIdx = doc_id2idx[docId]
            outcoming_docs =  self.index.docs[docId].get('outcoming_links', [])
            for out_docId in outcoming_docs:
                out_docIdx = doc_id2idx[out_docId]
                M[docIdx, out_docIdx] += 1
        return M



    def get_relevant_docs(self, query):
        base_relevant_docs = self.get_base_model_docs(query)
        outcoming_docs = self.get_docs_outcoming_docs(base_relevant_docs)
        if self.k>0:
            incoming_docs = self.get_docs_incoming_docs(base_relevant_docs, sample=True)
            return list(set(base_relevant_docs + outcoming_docs + incoming_docs))
        else:
            return list(set(base_relevant_docs + outcoming_docs))


    def get_base_model_docs(self, query):
        relevant_docs_ranks = self.base_model.getRanking(query)[:self.seeds]
        return [doc_rank[0] for doc_rank in relevant_docs_ranks]


    def get_docs_incoming_docs(self, docs, sample=False):
        incoming_docs = []
        for docId in docs:
            doc_incoming_docs = self.get_incoming_docs(docId)
            if sample:
                k_doc = min(len(doc_incoming_docs), self.k)
                doc_incoming_docs = random.sample(doc_incoming_docs, k_doc)
            incoming_docs.extend(doc_incoming_docs)
        return list(set(incoming_docs))

    def get_docs_outcoming_docs(self, docs):
        outcoming_docs = []
        [outcoming_docs.extend(self.get_outcoming_docs(docId)) for docId in docs]
        return list(set(outcoming_docs))


    def get_incoming_docs(self, docId, sample_from=None):
        return self.index.docs[docId].get('incoming_links', [])

    def get_outcoming_docs(self, docId, sample_from=None):
        return self.index.docs[docId].get('outcoming_links', [])



class PageRank(RandomWalk):
    def __init__(self, index, base_model=Okapi,
                  seeds=50, k=2, epsilon=10e-8):
        base_model = base_model(index)
        super(PageRank, self).__init__(index, base_model, seeds, k)
        self.epsilon = epsilon

    def getDocScores(self):

        mu = self.get_mu(self.get_docs_links_normalized())
        docsScores = {self.doc_idx2id[idx]: score  for idx, score in enumerate(mu)}
        ordered_scores = self.order_dict(docsScores)
        docsScores = {docId: rank for rank, (docId, _) in enumerate(ordered_scores)}
        return docsScores

    def get_mu(self, A, d=.85):

        N = len(A)
        C = (1.-d)/N
        last_mu = np.ones(N)*10000
        mu = (1./N)*np.ones(N)
        while np.linalg.norm(mu - last_mu, 2) > self.epsilon:
            last_mu = mu
            mu = C*np.ones(N) + d*A.dot(mu)
        return mu

    def getScores(self, query, d=.85):
        relevant_docs = self.get_relevant_docs(query)
        A = self.get_docs_links_normalized(relevant_docs)
        mu = self.get_mu(A, d)
        return {self.A_idx2id[idx]: score  for idx, score in enumerate(mu)}

    def get_docs_links_normalized(self, docs=None):
        A = self.docs_links.copy().T
        if docs is not None:
            docs_idx = [self.doc_id2idx[docId] for docId in docs]
            A = A[docs_idx][:,docs_idx]
            self.A_idx2id = {idx:docId for idx, docId in enumerate(docs)}
        A[:,A.sum(axis=0)==0] = 1
        A = A/A.sum(axis=0).astype(float)
        return A

class HITS(RandomWalk):
    def __init__(self, index, base_model=Okapi,
                 seeds=1000, k=20, epsilon=10e-5):
        #base_model = base_model(index)
        base_model = LanguageModel(index, lissage=.8)
        super(HITS, self).__init__(index, base_model, seeds, k)
        self.epsilon = epsilon

    def getDocScores(self):
        a = self.get_a(self.docs_links)
        docsScores = {self.doc_idx2id[idx]: score  for idx, score in enumerate(a)}
        ordered_scores = self.order_dict(docsScores)
        docsScores = {docId: rank for rank, (docId, _) in enumerate(ordered_scores)}
        return docsScores

    def get_links(self, docs):
        A = self.docs_links
        docs_idx = [self.doc_id2idx[docId] for docId in docs]
        A = A[docs_idx][:,docs_idx]
        self.A_idx2id = {idx:docId for idx, docId in enumerate(docs)}
        return A


    def getScores(self, query):
        relevant_docs = self.get_relevant_docs(query)
        a = self.get_a(self.get_links(relevant_docs))
        return {self.A_idx2id[idx]: score  for idx, score in enumerate(a)}


    def get_a(self, M):
        N = len(M)
        a = np.ones(N)
        h = np.ones(N)
        last_a = np.ones(N)*1000
        last_h = np.ones(N)*1000

        while np.linalg.norm(a - last_a, 2) > self.epsilon:
            next_a = np.dot(M.T, h)
            next_h = np.dot(M, a)

            next_a = next_a/np.linalg.norm(next_a, 2)
            next_h = next_h/np.linalg.norm(next_h, 2)

            last_a = a
            last_h = h

            a = next_a
            h = next_h
        return a

