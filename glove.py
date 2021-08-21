import numpy as np
from numpy.linalg import eig

sentences = [['chocolate','like','I'],
			['I','like','toffee']]


terms = []
#list to store the index terms present in the corpus
#N
#number of samples
sort_terms = []
dicti = {}
dicti_mean = {}
co_occur = []
covariance_matrix = []
embedding = []
#list to store the elements of co occurrence matrix

def index_terms():
    num = 0
    for lis in sentences:
        for word in lis:
            if word not in terms:
                dicti[num] = word
                num += 1
                terms.append(word)

    N = len(terms)
    
    return N
    #print(terms)
    #print(dicti)

#generating the co-occurrence matrix for the sentences
def co_occurrence_matrix():
    context = []
    count = 0
    
    sort_terms = sorted (terms)
    for i in range(len(terms)):
        for j in range(len(terms)):
            count = context_count(dicti[i],dicti[j])
            
            context.append(count)
        
        copy_lis = context.copy()        
        co_occur.append(copy_lis)
        context.clear()

    print(co_occur)

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
            if ( ind-1 >= 0  and lis[ind-1] == word2):
                count += 1

    return count

def compute_mean():
    num = 0
    for col in co_occur:
        avg = sum(col) / len(col)
        dicti_mean[dicti[num]] = avg
        num += 1

    #print(dicti_mean)

def covariance( ind_1 , ind_2, N ):
    #print(co_occur[ind_1],co_occur[ind_2])

    ans = 0
    
    
    mean_1 = dicti_mean[dicti[ind_1]]
    mean_2 = dicti_mean[dicti[ind_2]]

    for i in range(len(co_occur[ind_1])):
        ans += ( (co_occur[ind_1][i] - mean_1) * (co_occur[ind_2][i] - mean_2) )
        
        
    ans = ans/ (N-1)
    return ans

def covari_matrix(N):
    dummy = []
    cov = []
    for i in range(N):
        for j in range(N):
            ans = covariance( i,j,N )
            cov.append(ans)
        dummy = cov.copy()
        covariance_matrix.append(dummy)
        cov.clear()
    print (covariance_matrix )

def compute_eigen():
    a = np.array(covariance_matrix)
    E_value,E_vector=eig(a)

    return E_vector

def principle_component(E_vector):
    print(dicti,dicti_mean,co_occur)

    matrix_2 = []
    lis = []
    embed = []
    #len(co_occur)
    for i in range (len(co_occur)):
        num = 0
        for j in range (len(co_occur)):
            for k in range (len(co_occur)):
                ans = co_occur[k][j] - dicti_mean [dicti[k]]

                lis.append(ans)

            matrix_2 = np.array(lis)
            #matrix_2 = matrix_2.transpose()
            matrix_1 = E_vector[i]

            #print(matrix_1,matrix_2)

            result = np.dot(matrix_1,matrix_2.transpose())
            #print(result)
            embed.append(  result)
            dummy = embed.copy()
            lis.clear()
        embedding.append(dummy)
        embed.clear()
        #print(embed)
    print(embedding)
    for i in embedding:
        print(i,"\n")
if __name__=="__main__":
    
    N = index_terms()
    co_occurrence_matrix()
    compute_mean()
    covari_matrix(N)
    E_vector = compute_eigen()
    #print(E_vector)
    principle_component(E_vector)
    