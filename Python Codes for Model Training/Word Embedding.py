#Importing Necessary Libraries
import os
import re
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import FastText
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt')

#Cleaning datas for Azerbaijani language
files = list()
for file in os.listdir("path_to_data_files"):
    if file.endswith('.txt'):
        text = (open("path_to_data_files" + file, encoding = "utf-8")).read()
        text = text.replace(u'\xa0', u'').replace(u'\n', u' ').replace("Ģ", "ş").replace("Ġ", "i").replace(",", " ").replace(":", " ").lower()
        text = text.replace(u'...', u'.').replace(u'...', u'.').replace(u'..', u'.').replace(u".", u" . ")
        text = re.sub("[^qüertyuiopöğasdfghjklıəzxcvbnmçşi. \.]", "", text)
        files.append(text)

#Merging all cleaned texts into one data corpus
corpus = ' '.join(files)

#Counting Number of Tokens, our corpus has 76116387 tokens
i = 0
pattern = re.compile(r'\w+')
for sent in nltk.sent_tokenize(corpus):
    i = i + len(pattern.findall(sent))
print(i)

#Creating new cleaned data corpus
file = open("dataCorpus.txt", "w", encoding = "utf-8")
file.write(corpus)
file.close()

#Creating CBOW Word2vec model
modelCBOWw2v = Word2Vec(sentences = gensim.models.word2vec.LineSentence("path_to_data_corpus"))

#Checking cosine similarity between two words
modelCBOWw2v.similarity('first_word', 'second_word')

#Showing top 5 similar words to a given words with their cosie similarities
modelCBOWw2v.wv.most_similar("word", topn = 5)

#Checking Word Analogy
modelCBOWw2v.wv.most_similar(positive = ["first_positive_word", "second_positive_word"], negative = ["negative_word"], topn = 1)

#Checking syntactic and semantic, capital-country scores of the model, Intrinsic Evaluation
print (f" syntactic score: - {modelCBOWw2v.wv.evaluate_word_analogies('path_to_syntactic_inputs')[0]}")
print (f" semantic score: - {modelCBOWw2v.wv.evaluate_word_analogies('path_to_semantic_inputs')[0]}" ) 
print (f" capital-country score: - {modelCBOWw2v.wv.evaluate_word_analogies('path_to_capital_country_score_data')[0]}" )

#Saving The Model
modelCBOWw2v.wv.save_word2vec_format("CBOWw2v.txt", binary = False)

#Delete model in order not to load RAM a lot
modelCBOWw2v = None



#Creating Skip-Gram Word2vec model
modelSGw2v = Word2Vec(sentences = gensim.models.word2vec.LineSentence("path_to_data_corpus"), sg = 1)

#Checking cosine similarity between two words
modelSGw2v.similarity('first_word', 'second_word')

#Showing top 5 similar words to a given words with their cosie similarities
modelSGw2v.wv.most_similar("word", topn = 5)

#Checking Word Analogy
modelSGw2v.wv.most_similar(positive = ["first_positive_word", "second_positive_word"], negative = ["negative_word"], topn = 1)

#Checking syntactic and semantic, capital-country scores of the model, Intrinsic Evaluation
print (f" syntactic score: - {modelSGw2v.wv.evaluate_word_analogies('path_to_syntactic_inputs')[0]}")
print (f" semantic score: - {modelSGw2v.wv.evaluate_word_analogies('path_to_semantic_inputs')[0]}" ) 
print (f" capital-country score: - {modelSGw2v.wv.evaluate_word_analogies('path_to_capital_country_score_data')[0]}" )

#Saving The Model
modelSGw2v.wv.save_word2vec_format("SGw2v.txt", binary = False)

#Delete model in order not to load RAM a lot
modelSGw2v = None



#Creating CBOW FastText model
modelCBOWFT = FastText(sentences = gensim.models.word2vec.LineSentence("path_to_data_corpus"), min_n = 4, max_n = 2)

#Checking cosine similarity between two words
modelCBOWFT.similarity('first_word', 'second_word')

#Showing top 5 similar words to a given words with their cosie similarities
modelCBOWFT.wv.most_similar("word", topn = 5)

#Checking Word Analogy
modelCBOWFT.wv.most_similar(positive = ["first_positive_word", "second_positive_word"], negative = ["negative_word"], topn = 1)

#Checking syntactic and semantic, capital-country scores of the model, Intrinsic Evaluation
print (f" syntactic score: - {modelCBOWFT.wv.evaluate_word_analogies('path_to_syntactic_inputs')[0]}")
print (f" semantic score: - {modelCBOWFT.wv.evaluate_word_analogies('path_to_semantic_inputs')[0]}" ) 
print (f" capital-country score: - {modelCBOWFT.wv.evaluate_word_analogies('path_to_capital_country_score_data')[0]}" )

#Saving The Model
modelCBOWFT.wv.save_word2vec_format("CBOWFT.txt", binary = False)

#Delete model in order not to load RAM a lot
modelCBOWFT = None



#Creating Skip-Gram FastText model
modelSGFT = FastText(sentences=gensim.models.word2vec.LineSentence("/content/drive/MyDrive/corpus.txt"), sg = 1, min_n = 2, max_n = 4, size = 300, bucket = 500000)

#Checking cosine similarity between two words
modelSGFT.similarity('first_word', 'second_word')

#Showing top 5 similar words to a given words with their cosie similarities
modelSGFT.wv.most_similar("word", topn = 5)

#Checking Word Analogy
modelSGFT.wv.most_similar(positive = ["first_positive_word", "second_positive_word"], negative = ["negative_word"], topn = 1)

#Checking syntactic and semantic, capital-country scores of the model, Intrinsic Evaluation
print (f" syntactic score: - {modelSGFT.wv.evaluate_word_analogies('path_to_syntactic_inputs')[0]}")
print (f" semantic score: - {modelSGFT.wv.evaluate_word_analogies('path_to_semantic_inputs')[0]}" ) 
print (f" capital-country score: - {modelSGFT.wv.evaluate_word_analogies('path_to_capital_country_score_data')[0]}" )

#Saving The Model
modelSGFT.wv.save_word2vec_format("SGFT.txt", binary = False)

#Delete model in order not to load RAM a lot
modelSGFT = None

#After Creating and Saving all models, you can load and use each model with below statement
model_name = KeyedVectors.load_word2vec_format("path_to_model_file")
