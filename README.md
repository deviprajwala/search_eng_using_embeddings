# search_eng_using_embeddings
This repositary consists of the implementation of the search engine, initially we obtain the terms which are most similar to the words present in the query by making use of the 
embeddings of the words and compare them with the help of the cosine similarity measure.Once we a get a list of similar words we cluster the documents which contain those words and
display those documents for the given query.In this way we retrieve the documents when a query is given.

The repositary consists of the program for the search engine named search_eng_embed.py which contains the code for the search engine.The code is written in python. It also contains a 
csv file named corpus.csv which contains the name of the documents and the index terms present in them.We have two text files input.txt and output.txt which contains the input which is 
given to the program and the output which is redirected to a file.The output file contains the similar words corresponding to the query and a list of relevant documents.We have used the 
embeddings for words from a pre trained model from the website:https://nlp.stanford.edu/projects/glove/ ,it contains a 50 dimensional embeddings for around 400000 words.

The embeddings are obtained from the glove algorithm , glove stands for global vectors. Glove is an unsupervised learning algorithm for obtaining vector representations for words. 
Training is performed on aggregated global word-word co-occurrence statistics from a corpus.

Initially we import the basic packages like numpy, pandas and math for performing the basic operations.Then we read the data from the csv file as a dataframe, determine the number of 
rows and columns.Later read and separate the name of the documents and the terms present in it to a separate list  from the data frame and also create a dictionary which has the name of 
the document as key and the terms present in it as the list of strings  which is the value of the key.

Then we read the pre generated embeddings from a text file. Here we read each line from the text file and store the terms as keys and thier continous set of values i.e embeddings as a 
vector.Later we get the query input from the user, we have redirected it from the input.txt file.Then we split the query as a list of strings and get the list of embeddings given a 
query.

We then find the words in the text file which has similar meaning as that of the word present in the query, we get a list of words which have the similar meaning, then we find the 
cluster of documents which has the similar meaning or the similar embeddings as that of the words given in the query. In this way we get the relevant documents corresponding to the 
given query.
    
Refrences:
https://github.com/stanfordnlp/GloVe
https://nlp.stanford.edu/projects/glove/
https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010
https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/



#glove algorithm for generating embeddings

We obtain the set of similar words based on thier context.This algorithm is implemented and present in the file named glove.py in this repositary.The code is written in python.

Initially we compute the co occurence matrix for which we calculate covariance for each of the ordered pair , from which we generate the covariance matrix. After getting the covariance 
matrix we obtain the eigen vector later by making use of it we compute the priciple components.We append all these to form the embeddings matrix.

Once we get the embeddings we compute the similar words by comparing the similarity measure calculated based on the cosine similarity between the embeddings of the words.

Refrences:
https://www.solver.com/principal-components
https://builtin.com/data-science/step-step-explanation-principal-component-analysis
https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285
https://datamahadev.com/stanfords-glove-implementation-using-python/
https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
https://www.youtube.com/watch?v=MLaJbA82nzk
