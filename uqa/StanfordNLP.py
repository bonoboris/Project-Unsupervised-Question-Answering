import stanfordnlp as stanford
from nltk.tree import Tree
from stanfordnlp.server import CoreNLPClient
from stanfordcorenlp import StanfordCoreNLP

text = "Quand c'est fini, je chercherai les enfants de moins de 3 ans. Le vin produit en 1998 a notamment un goût très spécial."


# stanfordcorenlp
# nlp = StanfordCoreNLP(r'F:\EtudeCentraleSupelec\Projet OSY\stanford-corenlp-full-2018-10-05')
# # print ('Constituency Parsing:', nlp.parse(text))
# for i in nlp.parse(text):
# 	print(i)
# nlp.close()


# stanfordnlp.server
import os
os.environ["CORENLP_HOME"] = r'F:\EtudeCentraleSupelec\Projet OSY\stanford-corenlp-full-2018-10-05'
with CoreNLPClient(annotators=[ 'tokenize','ssplit','pos','parse','ner'],
                   timeout=30000,
                   output_format="json",
                   properties={'tokenize.language' :'fr',
                               'pos.model' : 'edu/stanford/nlp/models/pos-tagger/french/french-ud.tagger',
                               'parse.model' : 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz'}) as client :
	ann = client.annotate(text)
# {'sentences':
# 	[
# 		{'index':0, 'parse': "...", 'ner' :},
#		{'index':1, 'parse': "...", 'ner':},
#		...
#	]
# }


output = ann['sentences'][0]['parse']
parsetree = []
for sentence in ann['sentences']:
	parsetree.append(Tree.fromstring(sentence['parse']))



def getNodes(parent):
	ROOT = 'SENT'
	for node in parent:
		if type(node) is Tree:
			if node.label() == ROOT:
				print ("======== Sentence =========")
				print ("Sentence:", " ".join(node.leaves()), '\n')
			else:
				print ("Label:", node.label(), " --- Leaves:", node.leaves())
			getNodes(node)
		else:
			continue

for parse in parsetree:
	parse.pretty_print()
	getNodes(parse)