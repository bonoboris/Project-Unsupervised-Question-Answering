import gensim
import re
import numpy as np
from string import punctuation
from nltk.corpus import stopwords as nltk_stopwords
from src.textformatting import *
from src.spacywrapper import SpacyFrenchModelWrapper
from src.baselinemethods import preprocessor
from src.distances import mean_cosine

contracted_articles = {"l'", "d'", "s'", "j'", "t'", "m'", "n'", "t"}
stopwords = nltk_stopwords.words('french') + list(punctuation) + list(contracted_articles)

def isNumber(string):
    for s in string:
        if s not in '0123456789':
            return False
    return True

def word2vec(file_path, model_path):

    w2v_model = gensim.models.Word2Vec.load(model_path)

    json_content = read_json(file_path)
    contexts, questions = list(), list()
    for triplet in json_content["triplets"]:
        contexts.append(triplet["context"])
        questions.append(triplet["question"])

    return paragraphes2vec(w2v_model, contexts), paragraphes2vec(w2v_model, questions)


def paragraphes2vec(model, paragraphes):
    vecs_mat = [[] for i in range(len(paragraphes))]
    for index, para in enumerate(paragraphes):
        sentences = re.split("[.!?]", para)
        for sent in sentences:
            sent = preprocessor(sent)
            tokens = SpacyFrenchModelWrapper.stem(sent, stopwords)
            for token in tokens:
                if isNumber(token):
                    continue
                token = token.lower()
                try:
                    vecs_mat[index].append(np.array(model.wv[token]))
                except KeyError:
                    print("The word", token, "does not appear in this model")
    return vecs_mat


if __name__ == '__main__':
    model_path = '../models/frwiki.gensim'
    json_file_path = "../ressources/Jacques_Chirac_intro_manual.json"
    context2vec, question2vec = word2vec(json_file_path, model_path)
    print(context2vec[0], context2vec[0][0])
    for i in range(len(context2vec)):
        print(mean_cosine(context2vec[i], question2vec[i]))
