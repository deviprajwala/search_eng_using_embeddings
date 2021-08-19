import numpy as np
from numpy.linalg import eig

sentences = [['chocolate','like','I'],
			['I','like','toffee']]


terms = []
#list to store the index terms present in the corpus

sort_terms = []
dicti = {}

co_occur = []
#list to store the elements of co occurrence matrix

def index_terms():
    num = 0
    for lis in sentences:
        for word in lis:
            if word not in terms:
                dicti[num] = word
                num += 1
                terms.append(word)

    
    print(terms)
    print(dicti)

#generating the co-occurrence matrix for the sentences
def co_occurrence_matrix():
    context = []
    count = 0
    
    sort_terms = sorted (terms)
    for i in range(len(terms)):
        for j in range(len(terms)):
            count = context_count(dicti[i],dicti[j])
                '''for lis in sentences:

                    index = lis.index(dicti[i])
                    print (dicti[i],index,i,j,len(terms),"\n")
                    if( index+1 < len(terms) ):
                        if( lis[index+1] == dicti[j] ):
                            print( dicti[j] )
                        
                        
                        if( lis[index+1] == dicti[j] ):
                            context += 1
                        
                    if( index-1 >= 0 and lis[index-1] == dicti[j] ):
                        context += 1
                        print(dicti[index])

                    context = 0'''


        print("\n")

def context_count(word1,word2):
    #print(word1,word2)
    count = 0
    if( word1 == word2):
        return count

    for lis in sentences:
        if (word1 in lis):
            ind = lis.index(word1)
            if ( ind+1 < len(lis) and lis[ind+1] == word2):
                count += 1
                print(word1,word2)
            if ( ind-1 >= 0 ) and lis[ind-1] == word2):
                count += 1

    return count


if __name__=="__main__":
    
    index_terms()
    co_occurrence_matrix()