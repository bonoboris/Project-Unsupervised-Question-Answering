from os import path
import gensim
import re
import numpy as np
from string import punctuation
from nltk.corpus import stopwords as nltk_stopwords
from textformatting import *
from spacywrapper import SpacyFrenchModelWrapper
from baselinemethods import preprocessor
import distances
from data import DATA_PATH
from models import MODELS_PATH

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

    return embed(w2v_model, contexts), embed(w2v_model, questions)


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


def embed(model, docs):
    doc_mat = []
    tokenizer = re.compile(r"[a-zA-ZÀ-ÿ]{3,}")
    for doc in docs:
        words = tokenizer.findall(doc)
        embeddings = []
        for word in words:
            word = word.lower()
            try:
                embeddings.append(np.array(model.wv[word]))
            except KeyError:
                print("The word", word, "does not appear in this model")
        doc_mat.append(embeddings)
    return doc_mat


if __name__ == '__main__':
    model_path = path.join(MODELS_PATH, 'frwiki.gensim')
    json_file_path = path.join(DATA_PATH, "Jacques_Chirac_intro_manual.json")
    print("Loading word2vec model...")
    context2vec, question2vec = word2vec(json_file_path, model_path)
    print("Done !")
    # print(context2vec[0], context2vec[0][0])
    cost_mat = np.array([[distances.minimal_assignment_cosine(question, context) for context in context2vec]
                         for question in question2vec])
    print("Cost matrix:")
    print(cost_mat)
    print("Predictions:")
    print(np.argmin(cost_mat, axis=0))