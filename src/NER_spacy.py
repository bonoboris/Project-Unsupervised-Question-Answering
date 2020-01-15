import stanfordnlp as stanford
from nltk.tree import Tree

text = 'Cherche les enfants de moins de 3 ans.'
# with CoreNLPClient(annotators=[ 'tokenize','ssplit','pos','parse'],
#                    timeout=30000,
#                    output_format="json",
#                    properties={'tokenize.language' :'fr',
#                                'pos.model' : 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
#                                'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz'}) as client :
#     ann = client.annotate(text)

# stanford.download('fr')

config = {
	'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize_model_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt',
	'lemma_model_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd_lemmatizer.pt',
	'depparse_model_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd_parser.pt',
	'depparse_pretrain_path': 'D:/stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt'
}

nlp = stanford.Pipeline(**config)
doc = nlp(text)
# output = ann['sentences'][0]['parse']


for sent in doc.sentences:
    for word in sent.words:
        print('text: ' + word.text+ '    lemma: ' + word.lemma + '    upos: ' + word.upos + '    governor: ' + str(word.governor) + '    dependency relation: ' + word.dependency_relation)
doc_const_parsing = doc.sentences[0]['dependency_relation']
parsetree = Tree.fromstring(doc_const_parsing)
parsetree.pretty_print()
