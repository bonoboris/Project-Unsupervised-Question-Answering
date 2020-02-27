import spacy


class _LazySpacyNlpModel(object):
    """Lazy loading """

    def __init__(self, name):
        self.name = name
        self._model = None

    def __get__(self, obj, type_=None):  # pylint: disable=method-hidden
        if self._model is None:
            print(f"Loading Spacy model: {self.name}")
            self._model = spacy.load(self.name)
            print(f"{self.name} model loaded!")
            self.__get__ = self._getter

        return self._model

    def __set__(self, obj, value):
        """Will prevent setting instance-bound descriptor but will not prevent setting class-bound descriptor!"""
        print("Setting LazySpacyNlpModel object")
        raise AttributeError("Can't set attribute !")

    def _getter(self, obj, type_=None):
        return self._model


class SpacyFrenchModelWrapper(object):
    model: spacy.language.Language = _LazySpacyNlpModel("fr_core_news_md")

    @classmethod
    def tokenize(cls, sentence, stopwords=list()):
        return [token.text for token in cls.model(sentence) if token.text not in stopwords]

    @classmethod
    def stem(cls, sentence, stopwords=list()):
        return [token.lemma_ for token in cls.model(sentence) if token.text not in stopwords]

    @classmethod
    def ner(cls, sentence):
        return [token.tag_ for token in cls.model(sentence)]


def main():
    print(f'Tokens: {SpacyFrenchModelWrapper.tokenize("Bonjour, je suis une phrase avec un accent é et à et â!")}')
    print(f'Stems: {SpacyFrenchModelWrapper.stem("Bonjour, est-il possible pour Chirac!")}')


if __name__ == '__main__':
    main()
