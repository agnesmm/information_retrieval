'''
Created on 6 sept. 2016

@author: SL
'''
from Parser import Parser
from Document import Document


import numpy as np

#===============================================================================
# /**
#  *
#  * Format of input files :
#  * .I <id>
#  * .T
#  * <Title>
#  * .A <Author>
#  * .K
#  * <Keywords>
#  * .W
#  * <Text>
#  * .X
#  * <Links>
#  *
#  */
#===============================================================================
class ParserCACM(Parser):
    def __init__(self):
        '''
        Constructor
        '''
        Parser.__init__(self,".I")

    def getDocument(self,text):
        other={};
        modeT=False;
        modeA=False;
        modeK=False;
        modeW=False;
        modeX=False;
        info=""
        identifier=""
        author=""
        kw=""
        links=""
        title=""
        texte=""

        st=text.split("\n");
        s=""
        for s in st:
            if(s.startswith(".I")):
                identifier=s[3:]
                continue

            if(s.startswith(".")):
                if(modeW):
                    texte=info
                    info=""
                    modeW=False

                if(modeA):
                    author=info
                    info=""
                    modeA=False

                if(modeK):
                    kw=info;
                    info="";
                    modeK=False;

                if(modeT):
                    title=info;
                    info="";
                    modeT=False

                if(modeX):
                    other["links"]=links;
                    info="";
                    modeX=False;



            if(s.startswith(".W")):
                modeW=True;
                info=s[2:];
                continue;

            if(s.startswith(".A")):
                modeA=True;
                info=s[2:];
                continue;

            if(s.startswith(".K")):
                modeK=True;
                info=s[2:];
                continue;

            if(s.startswith(".T")):
                modeT=True;
                info=s[2:];
                continue;

            if(s.startswith(".X")):
                modeX=True
                continue;

            if(modeX):
                l=s.split("\t");
                if(l[0]!=identifier):
                    if(len(l[0])>0):
                        links+=l[0]+";";

                continue;

            if((modeK) or (modeW) or (modeA) or (modeT)):
                #print "add "+s
                info+=" "+s



        if(modeW):
            texte=info;
            info="";
            modeW=False;

        if(modeA):
            author=info;
            info="";
            modeA=False;

        if(modeK):
            kw=info;
            info="";
            modeK=False;

        if(modeX):
            other["links"]=links;
            info=""
            modeX=False;

        if(modeT):
            title=info;
            info="";
            modeT=False;

        other["title"]=title
        other["text"]=texte
        other["author"]=author
        other["keywords"]=kw

        doc=Document(identifier,title+" \n "+author+" \n "+kw+" \n "+texte,other);

        return doc

#===============================================================================

class Query(object):

    def __init__(self, identifier, text, relevants):
        self.identifier = identifier
        self.text = text
        self.relevants = relevants
        self.others = None

    def getId(self):
        return self.identifier

    def getText(self):
        return self.text

    def getRelevants(self):
        return self.relevants

    def get(self,key):
        return self.others[key]

    def set(self,key,value):
        self.others[key]=value

    def __str__(self):
        return "id="+str(self.identifier)+"\n"+str(self.text)+"\n"+str(self.relevants['doc_id'])


class QueryParser():
    def __init__(self, rel_filename, parser=ParserCACM()):
        self.q = parser
        self.rel = self.load_rel(rel_filename)

    def nextQuery(self):
        query = self.q.nextDocument()
        if query is None:
            return None
        return self.getQuery(query, self.rel)

    def load_rel(self, rel_filename):
        dtype=[('qry_id', np.int32), ('doc_id', np.int32),
          ('pertinence', np.float32), ('idSubtopic', np.float32)]

        return np.loadtxt(rel_filename, dtype=dtype)


    def getQuery(self, query, rel):
        query_id = int(query.getId())
        query_text = query.getText()
        query_idx = np.where(rel['qry_id']==query_id)
        return Query(query_id, query_text, rel[query_idx])


    def split_query_dataset(self, split=.8):
        queries = []
        while True:
            query = self.nextQuery()
            if query is None:
                break
            if len(query.relevants)==0:
                continue
            queries.append(query)

        train_size = int(split*len(queries))

        return queries[:train_size], queries[train_size:]




# rel_filename = 'cacm/cacm.rel'
# query_filename = 'cacm/cacm.qry'

# q = QueryParser(query_filename, rel_filename)
# print(q.nextQuery())
# print(q.nextQuery())
# print(q.nextQuery())




#===============================================================================
# a=ParserCACM()
# a.initFile("../../data/cacm/cacm.txt")
# x1=a.nextDocument()
# print x1
# x2=a.nextDocument()
# print x2
# x3=a.nextDocument()
# print x3
# x4=a.nextDocument()
# print x4
#===============================================================================
