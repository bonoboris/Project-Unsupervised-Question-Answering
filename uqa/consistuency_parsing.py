'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/
'''

from stanfordcorenlp import StanfordCoreNLP
import logging
import json
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
import os


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   lang='fr')  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'ssplit,pos,parse',
            'pipelineLanguage': 'fr',
            'outputFormat': 'json'
        }

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))


# The server needs to know the location of the following files:
#   - stanford-corenlp-X.X.X.jar
#   - stanford-corenlp-X.X.X-models.jar
STANFORD = os.path.join("models", "stanford-corenlp-full-2018-10-05")

# Create the server
server = CoreNLPServer(
    os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
    os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"),
    verbose=True,
    port=8005
)

# Start the server in the background
server.start()

if __name__ == '__main__':
    server.start()
    parser = CoreNLPParser()
    parse = next(parser.raw_parse("Le chien est dans la prairie"))
