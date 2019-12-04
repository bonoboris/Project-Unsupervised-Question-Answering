import os
import re
import gensim
import logging


def extract_sentences(TextPath):
    """
    Turns a collection of plain text files into a list of lists of word tokens.
    """
    print("--extract_sentences")
    Sentences = []
    for File in os.listdir(TextDir):
        with open(File, "r") as InFile:
            Text = InFile.read()
            Text = re.sub("\n", " ", Text)
            Text = re.sub("--", "", Text)
            Text = re.sub("\.\.\.", ".", Text)
            Text = Text.lower()
            SentencesOne = []
            Text = re.split("[.!?]", Text)
            for Sent in Text:
                Sent = re.split("\W", Sent)
                Sent = [Token for Token in Sent if Token]
                SentencesOne.append(Sent)
            Sentences.extend(SentencesOne)
    return Sentences


class MySentences(object):

    def __init__(self, dirname):
        self.__dirname = dirname

    @property
    def dirname(self):
        return self.__dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for para in open(os.path.join(self.dirname, fname)):
                if "<doc id" not in para and "</doc>" not in para and para != '\n':
                    sentences = re.split("[.!?]", para)
                    for sent in sentences:
                        if sent != '\n':
                            sent = re.split("\W", sent)
                            sent = [Token.lower() for Token in sent if Token]
                            sent = [Token for Token in sent if len(Token) > 2]
                            if len(sent) > 1:
                                yield sent


def build_model(directory, size, file):
    """
    Builds a word vector model of the text files given as input.
    This should be used for very large collections of text, as it is very memory-friendly.
    """
    print("--build_model_new")
    logging.basicConfig(filename="logging.txt", level=logging.INFO)

    sentences = MySentences(directory) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, min_count=10, size=size, workers=2)
    model.save(file)
    print("Done.")


if __name__ == '__main__':
    WorkDir = ""
    TextDir = WorkDir + "resource/"
    ModelFile = WorkDir + "chirac.gensim"
    Size = 100  # dimensions of the model
    build_model(TextDir, Size, ModelFile)

    model_path = 'chirac.gensim'
    w2v_model = gensim.models.Word2Vec.load(model_path)

    # torch_roi = torch.FloatTensor(w2v_model.wv['roi'])
    # torch_ordinateur = torch.FloatTensor(w2v_model.wv['ordinateur'])
    # print(torch.dot(torch_roi,torch_ordinateur))
    for i, word in enumerate(w2v_model.wv.vocab):
        if i == 10:
            break
        print(word)

    # try:
    #     print(w2v_model.wv.similarity('chirac','ministre'))
    # except KeyError:
    #     print("no this word")