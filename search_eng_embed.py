import numpy as np
import pandas 
import math
terms = []
#list to store the terms present in the documents
keys = []
#list to store the names of the documents
dicti = {}
#dictionary to store the name of the document and the terms present in it as a vector
dummy_list = []
#list for performing some operations and clearing them
embeddings_dict = {}

doc_average ={}
similar_embeddings = []
def read_embeddings():
    
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

    #print(dicti)
def vectorise():
    avg = 0
    for i in dicti:
        for words in dicti[i]:
            avg += sum(embeddings_dict[words])
        #print(avg, i)
        #print(embeddings_dict[words])
        doc_average.update({i:avg})
        avg = 0
       

def get_embedding(query):
    weight = 0
    for word in query:
        weight = sum(embeddings_dict[word])
        #print(embeddings_dict[word])
    return embeddings_dict[word]
    #sum or the average of embeddings can be used if the query is a sentence but here we are considering a single word as query

def cosine_similarity( list1, list2 ):
    numerator = 0
    denomi1 = 0
    denomi2 = 0

    for i in range (len(list1)):
        numerator += list1[i] * list2[i]
        denomi1 += list1[i] * list1[i]
        denomi2 += list2[i] * list2[i]
        
    denominator = math.sqrt(denomi1) * math.sqrt(denomi2)
    simili = numerator/denominator

    return simili

def similar_words(query_embeding):
    for i in embeddings_dict:
        sim = cosine_similarity(embeddings_dict[i],query_embeding)

        '''if(sim >= 0.8):
            print(i,sim)
        Here you can adjuct the similarity threshold and check for the similar words, the cosine similarity range is 0 to 1, in the above lines
        of code I have considered the similarity measure to be of minimum 0.8, it can be varied as per the requirement'''


        if( sim >= 0.8 ):
            similar_embeddings.append(i)

def clustering():
    print(similar_embeddings)
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

    read_embeddings()

    vectorise()

    print("Enter the query")
    query = input()
    #to get the query input from the user, the below input is given for obtaining the output as in output.txt file
    #one three three

    query=query.split(' ')
    #spliting the query as a list of strings
    
    query_weight = get_embedding(query)
    #print(query_weight)
    

    similar_words(query_weight)
    clustering()