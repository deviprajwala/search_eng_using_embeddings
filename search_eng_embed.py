#implementation of search engine by making use of the embeddings

import numpy as np
#module to perform mathematical operations on arrays

import pandas
#module to read the contents of the file from a csv file

import math
#module to perform mathematical functions

terms = []
#list to store the terms present in the documents

keys = []
#list to store the names of the documents

dicti = {}
#dictionary to store the name of the document as key and the terms present in it as a vector

dummy_list = []
#list for performing some operations and clearing them

embeddings_dict = {}
#dictionary to store the terms as keys and thier continous set of values i.e embeddings as a vector

#doc_average = {}
#dictionary to store the name of the document as key and the average of embeddings of terms present in it 

doc_cluster = []
#list to store the set of names of the documents belonging to same cluster

similar_embeddings = []
#list to store the set of terms which has similar meaning as that of the word given in the query

def read_embeddings():
    '''function to read the pre generated embeddings from a text file. Here we read each line from the text file and store the terms 
    as keys and thier continous set of values i.e embeddings as a vector '''
    
    with open("glove.6B.50d.txt", 'r') as f:
        #open the file in the read mode

        for line in f:
            values = line.split()
            word = values[0]
            #word from the text file

            vector = np.asarray(values[1:], "float32")
            #embeddings of the word

            embeddings_dict[word] = vector
            #storing it in the dictionary

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
    '''function to find the average weight of each of the documents present in the corpus '''
    avg = 0
    #initialising the variable to zero

    for i in dicti:
        for words in dicti[i]:
            avg += sum(embeddings_dict[words])
            #calculate the sum of the embeddings

        doc_average.update({i:avg/len(words)})
        #updating the dictionary

        avg = 0
        #reinitialising the variable to zero


def get_embedding(query):
    ''' function to get the list of embeddings given a certain query'''

    #weight = 0
    for word in query:
        weight = sum(embeddings_dict[word])
        #print(embeddings_dict[word])'''
    return embeddings_dict[word]
    #sum or the average of embeddings can be used if the query is a sentence but here we are considering a single word as query

def cosine_similarity( list1, list2 ):
    ''' function to calculate the similarity measure between the two words. the embeddings of the words are given as function parameters in the form 
    of the lists, the function return the similarity value which is computed and it ranges from 0 to 1'''

    numerator = 0
    denomi1 = 0
    denomi2 = 0
    #initialising the variables to zero which is further required for the computation

    for i in range (len(list1)):
        numerator += list1[i] * list2[i]
        denomi1 += list1[i] * list1[i]
        denomi2 += list2[i] * list2[i]
        
    denominator = math.sqrt(denomi1) * math.sqrt(denomi2)
    simili = numerator/denominator
    #computing the cosine similarity measure by making use of the embeddings

    return simili
    #returning the similarity measure computed

def similar_words(query_embeding):
    ''' function to find the words in the text file which has similar meaning as that of the word present in the query, we get a list of words
    which have the similar meaning '''

    for i in embeddings_dict:
        #for each of the words in the text file we compute the similarity measure with that of the word present in the query
        sim = cosine_similarity(embeddings_dict[i],query_embeding)

        '''if(sim >= 0.8):
            print(i,sim)
        Here you can adjuct the similarity threshold and check for the similar words, the cosine similarity range is 0 to 1, in the above lines
        of code I have considered the similarity measure to be of minimum 0.8, it can be varied as per the requirement'''


        if( sim >= 0.8 ):
            #if the computed similarity measure is greater than the certain value , in the above lines of code we have considered the limit to be
            #0.8, and those documents are appended to the list
            similar_embeddings.append(i)

    print("The set of similar words are:")
    print(similar_embeddings)


def clustering():
    '''function to find the cluster of documents which has the similar meaning or the similar embeddings as that of the words given in the query '''

    for doc in dicti:
        #for each of the documents present in the corpus we find the similar words for the retrieval

        for index_term in dicti[doc]:
            #for each term present in the document , we compare it with the similar words present in the list

            if index_term in similar_embeddings:
                #if similar terms are present , we group those documents and append it to the cluster
                doc_cluster.append(doc)

    print("The set of relevant documents are:")
    print(doc_cluster)
    #to display the list of the relevant documents corresponding to the given query


        
if __name__=="__main__":
    corpus = pandas.read_csv(r'corpus.csv')
    #to read the data from the csv file as a dataframe

    rows = len( corpus )
    #to get the number of rows

    cols = len( corpus.columns ) 
    #to get the number of columns

    filter( corpus, rows, cols )
    #function call to read and separate the name of the documents and the terms present in it to a separate list  from the data frame and also create a dictionary which 
    #has the name of the document as key and the terms present in it as the list of strings  which is the value of the key.

    read_embeddings()
    # function to read the pre generated embeddings from a text file. Here we read each line from the text file and store the terms as keys and 
    # thier continous set of values i.e embeddings as a vector.
    

    #vectorise()
    #function to find the average weight of each of the documents present in the corpus

    print("Enter the query")
    query = input()
    #to get the query input from the user, the below input is given for obtaining the output as in output.txt file
    

    query=query.split(' ')
    #spliting the query as a list of strings
    
    query_weight = get_embedding(query)
    #function to get the list of embeddings given a certain query
    #print(query_weight)
    
    similar_words(query_weight)
    #function to find the words in the text file which has similar meaning as that of the word present in the query, we get a list of words
    #which have the similar meaning

    clustering()
    #function to find the cluster of documents which has the similar meaning or the similar embeddings as that of the words given in the query
