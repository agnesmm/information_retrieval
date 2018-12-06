'''
Created on 6 sept. 2016

@author: SL
'''
import os
from Document import Document
class Parser(object):
    '''
    classdocs
    '''


    def __init__(self, begin, end=""):
        '''
        Constructor
        '''
        self.begin=begin
        self.end=end
        self.file=None


    def initFile(self,filename):
        self.file=open(filename,"rb")


    def __del__(self):
        if(self.file is not None):
            #print self.file.closed
            self.file.close()
            #print str(self.file.closed)

    def nextDocument(self):
        #if((self.file is None) or (self.file.closed())):
        #    return None
        d=None
        ligne=""
        ok=False;
        while(not ok):
            st=""
            read=False;
            start=0;
            nbBytes=0;
            while(True):
                curOff=self.file.tell();

                ligne=self.file.readline();
                #self.file.seek(curOff,0)
                #print ligne


                if(len(ligne)==0):
                    if((len(self.end)==0) and read):
                        #print ok
                        nbBytes=curOff-start;
                        read=False
                        ok=True
                    break
                if(ligne.startswith(self.begin)):
                    if((len(self.end)==0) and read):
                        nbBytes=curOff-start;
                        read=False;
                        ok=True;

                        self.file.seek(curOff)

                        break;
                    else:
                        read=True
                        start=curOff
                if(read):
                    st+=(ligne)
                if((len(self.end)>0) and (ligne.startswith(self.end))):
                    read=False
                    ok=True;
                    nbBytes=self.file.tell()-start;
                    break
            if (ok):
                source=os.path.abspath(self.file.name)+";"+str(start)+";"+str(nbBytes)
                #print source
                d=self.getDocument(st);
                d.set("from", source);
            else:
                self.file.close();
                return None

        return d

    def getDocument(self,text):
        raise NotImplementedError





 #==============================================================================
 # Format of input files :
 # <Document id=<id>>
 # <Text>
 # </Document>
 #==============================================================================

class ParserSimple(Parser):
    def __init__(self):
        '''
        Constructor
        '''
        Parser.__init__(self,"<Document","</Document")

    def getDocument(self,text):
        other={};
        st=text.split("\n");
        sti=st[0].split("=");
        x=sti[1][0:sti[1].find(">")];
        l=[st[i] for i in range(1,len(st)-2)]
        text=" ".join(l)+"\n";
        doc=Document(x,text,other);
        return doc;

#===============================================================================
#===============================================================================
# a=ParserSimple()
# a.initFile("../../data/testPourParserSimple")
# x1=a.nextDocument()
# print x1
# x2=a.nextDocument()
# print x2
# x3=a.nextDocument()
# print x3
# x4=a.nextDocument()
# print x4
#===============================================================================
#===============================================================================
