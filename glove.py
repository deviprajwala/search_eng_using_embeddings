#implementing the glove algorithm for generating the embeddings for the words

import numpy as np
#module to perform mathematical operations on arrays

from numpy.linalg import eig
#Module eig to obtain the eigen vector form the covariance matrix

import math
#module to perform mathematical functions

sentences = [['I','like','chocolate'],
			['I','like','toffee']]
#sentences for which the embeddings are generated based on thier context


terms = []
#list to store the index terms present in the corpus

#N
#number of samples

dicti = {}
#dictionary which contains the index as key and the word as the value

dicti_mean = {}
#dictionary which contains word as the key and the mean of co occurrences as the value

co_occur = []
#to store the elements of the co occrence matrix, it contains the co occurrence of the words based on the context and the window size

covariance_matrix = []
#to store the elements of the covariance matrix, it contains the covariance of the ordered pairs as the element in the matrix

embedding = []
#matrix to store the embedding values corresponding to each of the word

similar_words = []
#list to store the words which are considered to be similar based on thier context

def index_terms():
    '''function to get the index terms from the sentence and store it in a dictionary by associating a number with it, here we get the number of terms
    present in the sentences '''

    num = 0
    #initialising the variable to 0

    for lis in sentences:
        for word in lis:
            #for each of the word present in the list of the sentences

            if word not in terms:
                #if the word is already present in the list of terms then we continue oe elese we append the word to the list of terms and add the word to the dictionary
                dicti[num] = word
                num += 1
                terms.append(word)

    N = len(terms)
    #here N is number of the index terms
    
    return N
    #reture the value of N


def co_occurrence_matrix():
    '''function to generate the co occurence matrix based on the context of the words '''

    context = []
    #list to store the count of co occurrence based on the context window( here we have kept window size as 1)

    count = 0
    #initialising the variable to zero

    for i in range(len(terms)):
        for j in range(len(terms)):
            count = context_count(dicti[i],dicti[j])
            #to get the count of the co occurrence

            context.append(count)
            #append the count to the list
        
        copy_lis = context.copy()  
        #copying the list

        co_occur.append(copy_lis)
        #append the list to the co occurrence matrix

        context.clear()
        #clear the list

    #print(co_occur)

def context_count(word1,word2):
    ''' function to compute the count of the co occurrence of the words based on the index by considering the window size'''

    count = 0
    #initialising the count variable to zero

    if( word1 == word2):
        return count

    for lis in sentences:
        if (word1 in lis):
            ind = lis.index(word1)

            if ( ind+1 < len(lis) and lis[ind+1] == word2):
                #to check the context word towards the right with window size of one

                count += 1
                #increase the count variable by one

            if ( ind-1 >= 0  and lis[ind-1] == word2):
                #to check the context word towards the left with window size of one

                count += 1
                #increase the count variable by one

    return count
    #return the count variable to the calling function

def compute_mean():
    ''' function to comput the mean of the co occurrence matrix for each of the index words and then we store them in a dictionary with index word
    as the key and it's corresponding mean as the value in the dictionary '''

    num = 0
    #initialising the variable to zero

    for col in co_occur:
        avg = sum(col) / len(col)
        #calculating the average 

        dicti_mean[dicti[num]] = avg
        #updating the dictionary with the word index as the key and its mean as the value

        num += 1
        #increment the num variable by one

    #print(dicti_mean)

def covariance( ind_1 , ind_2, N ):
    ''' function to calculate the covariance of a ordered pair given thier values as the list, he we return the covariance computed.'''

    ans = 0
    #initialising the variable to zero
    
    mean_1 = dicti_mean[dicti[ind_1]]
    mean_2 = dicti_mean[dicti[ind_2]]
    #getting the mean value of each value in the pair

    for i in range(len(co_occur[ind_1])):
        ans += ( (co_occur[ind_1][i] - mean_1) * (co_occur[ind_2][i] - mean_2) )
        #operation performed to compute the covariance
        
    ans = ans/ (N-1)
    #covariance is computed

    return ans
    #computed covariance is returned back to the calling function

def covari_matrix(N):
    ''' function to get the covariance and append it in the proper place in the covariance matrix, here we get the covariance for all the
    possible pairs and append it to the matrix'''

    dummy = []
    cov = []
    #declaring the list which is required for the further operation

    for i in range(N):
        for j in range(N):
            ans = covariance( i,j,N )
            #calling the function to get the covariance

            cov.append(ans)
            #appending the ans to the list

        dummy = cov.copy()
        #copying the list to a dummy list

        covariance_matrix.append(dummy)
        #appending the list to form a matrix

        cov.clear()
        #clearing the list

    #print (covariance_matrix )

def compute_eigen():
    ''' function to compute the eigen vector'''

    a = np.array(covariance_matrix)
    E_value,E_vector=eig(a)
    #using the pre defined library to get the eigen value and vector

    return E_vector
    #return the eigen vector to the calling function

def principle_component(E_vector):
    ''' function to compute the pricipal component by making use of the eigen vector and store it in the embedding matrix'''

    matrix_2 = []
    lis = []
    embed = []
    #initialising the list which is required for the further computation

    for i in range (len(co_occur)):
        num = 0
        for j in range (len(co_occur)):
            for k in range (len(co_occur)):
                ans = co_occur[k][j] - dicti_mean [dicti[k]]
                #subtracting the co occurrence value with its mean value

                lis.append(ans)
                #appending the ans to the list

            matrix_2 = np.array(lis)
            #getting one matrix to perform the dot product

            matrix_1 = E_vector[i]
            #getting the matrix of eigen vector to perform the dot product

            result = np.dot(matrix_1,matrix_2.transpose())
            #matrix multiplication between matrix 1 and the transpose of matrix 2

            embed.append( result )
            #result is appended to the list

            dummy = embed.copy()
            #list is copied

            lis.clear()
            #list is cleared

        embedding.append(dummy)
        #a list is appended to the embedding

        embed.clear()
        #list is cleared

def find_similar(word):
    '''function to calculate the similar word for a given based on the context , we make use of cosine similarity measure to compute the 
    similairity and we have set the threshold limit to 0.8 '''
    
    numerator = 0
    denomi1 = 0
    denomi2 = 0
    #initialising the variables to zero , which is further required to perform computation

    val_list = list(dicti.values())
    #getting the list of values in the dictionary

    pos = val_list.index(word)
    #getting the index associated with the word

    for i in range(len(embedding)):
        for j in range(len(embedding)):
           #operations which are involved in computing the cosine similarity 
           
            numerator += embedding[j][pos] * embedding[j][i]
            denomi1 += embedding[j][pos] * embedding[j][pos]
            denomi2 += embedding[j][i] * embedding[j][i]


        denominator = math.sqrt(denomi1) * math.sqrt(denomi2)
        simili = numerator/denominator
        #similarity measure is computed 

        numerator = 0
        denomi1 = 0
        denomi2 = 0
        #reinitialising the variables to zero , which is further required to perform computation

        if(simili >= 0.8 and i != pos):
            #if the similarity calculated is greater than the threshold and if the word itself is the word in the function argument,append it to list
            similar_words.append(i)

    print("The word similar to ", word, "is :")
    for i in similar_words:
        print(dicti[i])
        #print the similar words by the index associated with them



if __name__=="__main__":
    
    N = index_terms()
    #function to get the index terms from the sentence and store it in a dictionary by associating a number with it, here we get the number of 
    #terms present in the sentences 

    co_occurrence_matrix()
    #function to generate the co occurence matrix based on the context of the words

    compute_mean()
    #function to comput the mean of the co occurrence matrix for each of the index words and then we store them in a dictionary with index word
    #as the key and it's corresponding mean as the value in the dictionary

    covari_matrix(N)
    #function to get the covariance and append it in the proper place in the covariance matrix, here we get the covariance for all the
    #possible pairs and append it to the matrix

    E_vector = compute_eigen()
    #function to compute the eigen vector

    principle_component(E_vector)
    #function to compute the pricipal component by making use of the eigen vector and store it in the embedding matrix

    print("Enter the word for which similar word has to be found:")
    word = input()
    #gettung the word for which similar word has to be found

    find_similar(word)
    # function to calculate the similar word for a given based on the context , we make use of cosine similarity measure to compute the 
    #similairity and we have set the threshold limit to 

    ''' if the input word chocolate is given the the word similar to that will be obtained as the toffee based the context, 
        
        Enter the word for which similar word has to be found:
        chocolate
        The word similar to  chocolate is :
        toffee

    '''