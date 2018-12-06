from evaluation import EvalMeasure, IRList, P, AP, EvalIRModel
from index import Index
from modeles import Vectoriel, LanguageModel, Okapi
from modeles import PageRank, HITS
from ParserCACM import ParserCACM, QueryParser
from TextRepresenter import PorterStemmer
from weighter import Weighter, BasicWeighter, SimpleWeighter, ThirdWeighter, FourthWeighter

import pprint
from tqdm import tqdm
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def order_dict(dico):
    return sorted(dico.items(),
      key=operator.itemgetter(1),
      reverse=True)


def compare_models(names, models, measures=['PrecisionRecall', 'AveragePrecision']):
    rel_filename = 'cacm/cacm.rel'
    query_filename = 'cacm/cacm.qry'
    query_parser = QueryParser(rel_filename)
    query_parser.q.initFile(query_filename)

    train_queries, _ = query_parser.split_query_dataset()

    pp = pprint.PrettyPrinter()

    models_scores = {}

    queries_results = {name: [] for name in names}

    for query in train_queries:
        query_txt = query.getText()
        for name, model in zip(names, models):
            scores = model.getRanking(query_txt)
            results = IRList(query, scores)
            queries_results[name].append(results)

    for model_name in names:
        print('==== {} ===='.format(model_name))
        model_score = EvalIRModel(measures).eval_model(queries_results[model_name])
        models_scores[model_name] = model_score

    pp.pprint(models_scores)
    return models_scores



def optimize_language_parameters(measures=['AveragePrecision']):

    index = Index()
    models, names = [], []
    lissage_values = np.arange(0.1, 1.0, 0.1)

    for l in lissage_values:
        models.append(LanguageModel(index, l))
        names.append('LanguageModel_'+str(l))

    scores = compare_models(names, models, measures)
    ap = [scores[model_name]['AveragePrecision']['mean'] for model_name in names]
    ap_std = [scores[model_name]['AveragePrecision']['eval_std'] for model_name in names]

    fig = plt.figure(figsize=(10,8))
    plt.plot(lissage_values, ap)
    plt.title('Language model : average precision depending on lissage values')
    plt.xlabel('Lissage')
    plt.ylabel('Average Precision')
    plt.savefig('plot/Language_Model_ap.png')

    fig = plt.figure(figsize=(10,8))
    plt.plot(lissage_values, ap_std)
    plt.title('Language model : average precision std depending on lissage values')
    plt.xlabel('Lissage')
    plt.ylabel('Average Precision')
    plt.savefig('plot/Language_Model_ap_std.png')

    best_model = names[np.argmax(ap)]
    print('best_model', best_model)

def optimize_okapi_parameters(measures=['AveragePrecision']):

    index = Index()
    k1_values = np.arange(1, 2.1, 0.1)
    b_values = np.arange(0.65, 0.90, 0.05)

    # k1_values = [1.1, 1.5]
    # b_values = [.7, .8]

    fig = plt.figure(figsize=(10,8))
    scores, names = {}, []
    colors=iter(cm.hot(np.linspace(0,.8,len(b_values))))
    for b in b_values:
        models_b, names_b = [], []
        for k1 in k1_values:
            name = 'Okapi_k1=%.2f_b=%.2f'%(k1, b)
            models_b.append(Okapi(index, k1, b))
            names_b.append(name)
            names.append(name)

        scores_b = compare_models(names_b, models_b, measures)
        scores.update(scores_b)
        ap_b = [scores_b[model_name]['AveragePrecision']['eval_std'] for model_name in names_b]


        color = next(colors)
        label = 'b = '+str(b)
        plt.plot(k1_values, ap_b, c=color, label=label)

    plt.title('Okapi model : average precision std depending on b values')
    plt.xlabel('k1')
    plt.ylabel('Average Precision std')
    plt.legend()
    plt.savefig('plot/Okapi_Model_ap_std.png')


    ap = [scores[model_name]['AveragePrecision']['mean'] for model_name in names]
    best_model = names[np.argmax(ap)]
    print('best_model', best_model)


def optimize_pagerank_parameters(measures=['AveragePrecision']):

    index = Index()

    models, names, scores = [], [], {}
    seeds_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
    k_values = [0, 1, 3, 5, 10, 20]

    colors=iter(cm.hot(np.linspace(0,.7,len(k_values))))
    fig = plt.figure(figsize=(10,8))
    for k in k_values:
        models_b, names_b = [], []
        for seeds in seeds_values:
            print('k : ', k, 'seeds : ', seeds)
            name = 'PageRank_seeds=%d_k=%d'%(seeds, k)
            models_b.append(PageRank(index, seeds=seeds, k=k))
            names_b.append(name)
            names.append(name)

        scores_b = compare_models(names_b, models_b, measures)
        scores.update(scores_b)
        ap_b = [scores_b[model_name]['AveragePrecision']['eval_std'] for model_name in names_b]


        color = next(colors)
        label = 'k = '+str(k)
        plt.plot(seeds_values, ap_b, c=color, label=label)
    plt.title('PageRank model : average precision std depending on k and seeds values')
    plt.xlabel('seeds')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.savefig('plot/PageRank_Model_ap_std.png')


def optimize_hits_parameters(measures=['AveragePrecision']):

    index = Index()


    models, names = [], []
    seeds_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
    k_values = [5]

    # seeds_values = [10, 100]
    # k_values = [5, 50]
    scores, names = {}, []

    fig = plt.figure(figsize=(10,8))
    colors=iter(cm.hot(np.linspace(0,.7,len(k_values))))
    for k in k_values:
        models_b, names_b = [], []
        for seeds in seeds_values:
            print('k : ', k, 'seeds : ', seeds)
            name = 'HITS_seeds=%d_k=%d'%(seeds, k)
            models_b.append(HITS(index, seeds=seeds, k=k))

            names_b.append(name)
            names.append(name)

        scores_b = compare_models(names_b, models_b, measures)
        scores.update(scores_b)
        ap_b = [scores_b[model_name]['AveragePrecision']['eval_std'] for model_name in names_b]


        color = next(colors)
        label = 'k = '+str(k)
        plt.plot(seeds_values, ap_b, c=color, label=label)
    plt.title('HITS model : average precision std depending on seeds values')
    plt.xlabel('seeds')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.savefig('plot/HITS_Model_ap_std.png')

def compare_vectoriel_models():
    index = Index()
    fig = plt.figure(figsize=(10,8))
    for c, normalized in zip(['red', 'blue'], [False, True]):
        models = [
          Vectoriel(index, BasicWeighter(index), normalized=normalized),
          Vectoriel(index, SimpleWeighter(index), normalized=normalized),
          Vectoriel(index, ThirdWeighter(index), normalized=normalized),
          Vectoriel(index, FourthWeighter(index), normalized=normalized),
          ]

        names = ['BasicWeighter', 'SimpleWeighter', 'ThirdWeighter', 'FourthWeighter']
        models_scores = compare_models(names, models)

        scores = {name: score['AveragePrecision']['mean'] for name, score in models_scores.items()}
        x = list(range(len(scores.values())))
        plt.bar(x, scores.values(), alpha=.4, color=c)
        plt.xticks(x, scores.keys())
        plt.title('Average Precision depending on weighter')

    plt.legend(['False', 'True'])
    plt.savefig('plot/vectoriel_models.png')

    def compare_model():
      index = Index()
      models = [Vectoriel(index, SimpleWeighter(index), normalized=True),
                  Vectoriel(index, BasicWeighter(index), normalized=True),
                  Vectoriel(index, ThirdWeighter(index), normalized=False),
                  Vectoriel(index, FourthWeighter(index), normalized=False),
                  Okapi(index),
                  LanguageModel(index, .7),
                  HITS(index, seeds=3, k=5),
                  PageRank(index, seeds=3, k=5)
                  ]
        names = ['Vectoriel1', 'Vectoriel2','Vectoriel3','Vectoriel4','Okapi', 'Language', 'HITS', 'PageRank']
        models_scores = compare_models(names, models)
        fig = plt.figure(figsize=(10,8))
        colors=iter(cm.hot(np.linspace(0,.7,len(names))))
        x = [s['AveragePrecision']['mean'] for s in scores.values()]
        plt.xticks(x, scores.keys())
        plt.bar(x, models_scores.values(), alpha=.4, color=c)
        # for n, s in models_scores.items():
        #     color = next(colors)
        #     y = s['PrecisionRecall']['mean']
        #     x = list(range(len(y)))
        #     plt.plot(x, y, color=color, label=n)
        # plt.legend()
        plt.savefig('plot/models_comparison_precision_eval.png')

if __name__=='__main__':
    # optimize_language_parameters()
    # optimize_okapi_parameters()
    # optimize_pagerank_parameters()
    # optimize_hits_parameters()
















