'''
Created on 5 sept. 2016

@author: SL
'''


class Document(object):
    '''
    classdocs
    '''


    def __init__(self,identifier,text="",others=None):
        '''
        Constructor
        '''
        self.identifier=identifier
        self.text=text
        self.others=others;
        
    def getId(self):
        return self.identifier
    
    def getText(self):
        return self.text
       
    def get(self,key):
        return self.others[key]  
    
    def set(self,key,value):
        self.others[key]=value

    def __str__(self):
        return "id="+self.identifier+"\n"+self.text+"\n"+str(self.others)




