from os import path
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sacremoses import MosesTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from spacywrapper import SpacyFrenchModelWrapper
from string import punctuation
from textformatting import read_json
import data

contracted_articles = {"l'", "d'", "s'", "j'", "t'", "m'", "n'"}
stopwords = nltk_stopwords.words('french') + list(punctuation) + list(contracted_articles)

DATA_PATH = path.dirname(data.__file__)


def preprocessor(string):
    """Preprocess and return a string."""
    string.replace("’", "'")
    return string


def nltk_tokenize_fr(text):
    return word_tokenize(text, language="french")


def moses_tokenize_fr(text):
    mose_tokenizer = MosesTokenizer("fr")
    return mose_tokenizer.tokenize(text, escape=False)


def filter_stopwords(words):
    return [word for word in words if word not in stopwords and not word.isdigit()]


def snowball_stem_fr(words):
    stemmer = SnowballStemmer("french")
    return [stemmer.stem(word) for word in words]


def test_preprocessing(text):
    nltk_words = nltk_tokenize_fr(text)
    print(f"Tokens (nltk):\n\t{nltk_words}")

    moses_words = moses_tokenize_fr(text)
    print(f"Tokens (moses):\n\t{moses_words}")

    nltk_words = filter_stopwords(nltk_words)
    print(f"Tokens, no stop words (nltk):\n\t{nltk_words}")

    moses_words = filter_stopwords(moses_words)
    print(f"Tokens, no stop words (moses):\n\t{moses_words}")

    nltk_stems = snowball_stem_fr(nltk_words)
    print(f"Stems (nltk):\n\t{nltk_stems}")
    moses_stems = snowball_stem_fr(moses_words)
    print(f"Stems (moses):\n\t{moses_stems}")

    spacy_stems = SpacyFrenchModelWrapper.stem(text)
    print(f"Stems (spacy):\n\t{spacy_stems}")


def compute_tfidf_mat(contexts, questions, tokenizer):
    tfidf = TfidfVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        analyzer="word")

    contexts_tfidf_mat = tfidf.fit_transform(contexts)
    questions_tfidf_mat = tfidf.transform(questions)

    return contexts_tfidf_mat, questions_tfidf_mat, tfidf.get_feature_names()


def match_questions_to_contexts(contexts_tfidf_mat, questions_tfidf_mat, metric="cosine"):
    if (metric not in {"cosine", "euclidian"}):
        raise ValueError("metric argument must be either: 'euclidian' or 'cosine")
    if metric == "cosine":
        distances = pairwise.cosine_distances(questions_tfidf_mat, contexts_tfidf_mat)
    else:
        distances = pairwise.euclidean_distances(questions_tfidf_mat, contexts_tfidf_mat)
    return [np.argmin(row) for row in distances]


def main():
    json_file_path = path.join(DATA_PATH, "Jacques_Chirac_intro_manual.json")
    json_content = read_json(json_file_path)

    contexts, questions = list(), list()
    for triplet in json_content["triplets"]:
        contexts.append(triplet["context"])
        questions.append(triplet["question"])

    spacy_res = compute_tfidf_mat(
        contexts, questions,
        lambda sentence: SpacyFrenchModelWrapper.stem(sentence, stopwords)
    )
    spacy_contexts_tfidf_mat, spacy_questions_tfidf_mat, spacy_tfidf_features_names = spacy_res
    print(match_questions_to_contexts(spacy_contexts_tfidf_mat, spacy_questions_tfidf_mat))


if __name__ == "__main__":
    # test_preprocessing("Bonjour, chat!")
    main()
