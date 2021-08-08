import numpy as np
import pandas 

terms = []
#list to store the terms present in the documents
keys = []
#list to store the names of the documents
dicti = {}
#dictionary to store the name of the document and the terms present in it as a vector
dummy_list = []
#list for performing some operations and clearing them

def read_embeddings():
    embeddings_dict = {}
    with open("glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    #print(embeddings_dict.keys())

def filter( documents, rows, cols ):
    '''function to read and separate the name of the documents and the terms present in it to a separate list  from the data frame and also create a dictionary which 
    has the name of the document as key and the terms present in it as the list of strings  which is the value of the key'''
    
    for i in range( rows ):
        for j in range( cols ):
            #traversal through the data frame

            if( j == 0 ):
                #first column has the name of the document in the csv file
                keys.append( documents.loc[i].iat[j] )
            else:
                dummy_list.append( documents.loc[i].iat[j] )
                #dummy list to update the terms in the dictionary

                if documents.loc[i].iat[j] not in terms:
                    #add the terms to the list if it is not present else continue
                    terms.append( documents.loc[i].iat[j] )

                
        copy = dummy_list.copy()
        #copying the the dummy list to a different list

        dicti.update( { documents.loc[i].iat[0]:copy } )
        #adding the key value pair to a dictionary

        dummy_list.clear()
        #clearing the dummy list

    print(dicti)

if __name__=="__main__":
    
    #print("hi")
    corpus = pandas.read_csv(r'corpus.csv')
    #to read the data from the csv file as a dataframe

    rows = len( corpus )
    #to get the number of rows

    cols = len( corpus.columns ) 
    #to get the number of columns

    filter( corpus, rows, cols )
    #function call to read and separate the name of the documents and the terms present in it to a separate list  from the data frame and also create a dictionary which 
    #has the name of the document as key and the terms present in it as the list of strings  which is the value of the key

    #print("hi")
    read_embeddings()