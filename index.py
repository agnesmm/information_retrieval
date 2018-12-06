from ParserCACM import ParserCACM
from TextRepresenter import PorterStemmer


import abc
import ast
import json
from os import path
import math
import numpy as np
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter()


class Index(object):
    def __init__(self, parser=ParserCACM,
                 textRepresenter=PorterStemmer,
                 create_index=False, directory='data/cacm',
                 txt_file='data/cacm/cacm.txt'):

        self.parser = parser()
        self.textRepresenter = textRepresenter()
        self.directory = directory

        self.txt = txt_file
        self.stems_filename = path.join(directory, 'stems.json')
        self.docs_filename =path.join(directory, 'docs.json')
        self.index_filename = path.join(directory, 'index.txt')
        self.inverted_index_filename = path.join(directory, 'inverted_index.txt')

        if create_index:
            self.create_index()

        else:
            self.stems = self.load_stems()
            self.docs = self.load_docs()

        self.corpus_length = self.get_corpus_length()
        self.documents_count = self.get_documents_count()


    def get_corpus_length(self):
        corpus_length = 0.0

        for docId in self.docs.keys():
            corpus_length += self.docs[docId]['doc_length']
        return corpus_length

    def get_documents_count(self):
        return len(self.docs)

    def load_stems(self):
        with open(self.stems_filename, 'r') as f:
            return json.load(f)

    def load_docs(self):
        with open(self.docs_filename, 'r') as f:
            return json.load(f)

    def getTfsForDoc(self, docId):
        docs = self.docs
        with open(self.index_filename, 'r') as f:
            f.seek(docs[docId]['position'])
            tfs_for_doc = f.read(docs[docId]['length'])
        return ast.literal_eval(tfs_for_doc)

    def getDocsLength(self, docId):
        return self.docs[docId]['doc_length']

    def getTfsForStem(self, stem):
        stems = self.stems
        if stem in stems.keys():
            with open(self.inverted_index_filename, 'r') as f:
                f.seek(stems[stem]['position'])
                tfs_for_doc = f.read(stems[stem]['length'])
        else:
            return {}
        return self.string2json(tfs_for_doc)


    def getIdfForStem(self, stem):
        df = float(len(self.getTfsForStem(stem)))
        N = float(self.documents_count)
        return math.log((1.+N)/(1.+df))

    # refaire marche pas (position cest pas ca cest index)
    def getStrDoc(self, docId):
        doc_rep_position = self.docs[docId]['position']
        doc_rep_length = self.docs[docId]['length']
        with open(self.txt, 'r') as f:
            f.seek(doc_rep_position)
            doc_str = f.read(doc_rep_length)
        return doc_str

    # refaire marche pas
    def getRepDoc(self, docId):
        doc_rep_position = self.docs[docId]['position']
        doc_rep_length = self.docs[docId]['length']
        with open(self.txt, 'r') as f:
            f.seek(doc_rep_position)
            doc_str = f.read(doc_rep_length)
        return self.parser.getDocument(doc_str)




    def create_index(self):

        textRepresenter = self.textRepresenter
        parser = self.parser
        parser.initFile(self.txt)

        index = ""
        docs, stems = {}, {}
        while True:
        # i=0
        # while i < 20:
        #     i += 1

            doc = parser.nextDocument()
            if doc is None:
              break

            doc_id = doc.getId()
            doc_rep = textRepresenter.getTextRepresentation(doc.getText())
            doc_rep_str = str(doc_rep)
            links = doc.others.get('links', '')
            doc_outcoming_links = links.split(';')[:-1]

            # index
            docs[doc_id] = docs.get(doc_id, {})
            docs[doc_id].update({"position": len(index),
                                "length": len(doc_rep_str),
                                "doc_length": sum(doc_rep.values()),
                                "outcoming_links": doc_outcoming_links})

            for d in doc_outcoming_links:
                docs[d] = docs.get(d, {})
                incoming_links = docs[d].get('incoming_links', [])
                incoming_links.append(doc_id)
                docs[d]['incoming_links'] = incoming_links


            index += doc_rep_str

            # intermediate inverted index
            for stem, stem_count in doc_rep.items():
                stem_rep_len = len(str({doc_id: stem_count}))
                if stem in stems.keys():
                    stems[stem]["length"] += stem_rep_len
                else:
                    stems[stem] = {"length": stem_rep_len}

            stem_position = 0
            for stem, stem_info in stems.items():
                stems[stem]["position"] = stem_position
                stems[stem]["writing_position"] = stem_position
                stem_position += stem_info["length"]



        inverted_index = "x"*stem_position

        with open(self.index_filename, 'w') as f:
            f.write(index)

        with open(self.inverted_index_filename, 'w') as f:
            f.write(inverted_index)

        parser = self.parser
        parser.initFile(self.txt)

        with open(self.inverted_index_filename, 'w') as f:
            while True:
                doc = parser.nextDocument()
                if doc is None:
                    break
                doc_id = doc.getId()
                doc_rep = textRepresenter.getTextRepresentation(doc.getText())

                for stem, stem_freq in doc_rep.items():
                    stem_position = stems[stem]["writing_position"]
                    stem_rep = str({doc_id: stem_freq})

                    f.seek(stem_position)
                    f.write(stem_rep)

                    stems[stem]["writing_position"] = stems[stem]["writing_position"]+len(stem_rep)



        self.docs = docs
        self.stems = stems

        with open(self.stems_filename, 'w') as f:
            json.dump(stems, f)
        with open(self.docs_filename, 'w') as f:
            json.dump(docs, f)

        return

    def string2json(self, string):
        '''convert "{'4': 1}{'7': 10}" to {'4':1, '7':10} '''
        string_list = [s+'}' for s in string.split('}')][:-1]
        json_list = [ast.literal_eval(s) for s in string_list]
        return {d.keys()[0]: d.values()[0] for d in json_list}



if __name__ == '__main__':
    parser = ParserCACM()
    parser.initFile("cacm/cacm.txt")


    index = Index(name='test', parser=ParserCACM,
                  textRepresenter=PorterStemmer, create_index=False)

    for i in range(20):
        doc = parser.nextDocument()
    doc_id = doc.getId()
    doc_rep = PorterStemmer().getTextRepresentation(doc.getText())

    print('doc_rep', doc_rep)
    print('length', index.getDocsLength(doc_id))

    tfs_for_doc = index.getTfsForDoc(doc_id)
    print('tfs_for_doc', tfs_for_doc)

    tfs_for_stem = index.getTfsForStem('diverg')
    print('tfs_for_stem diverg', tfs_for_stem)




