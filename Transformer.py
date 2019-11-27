import torch
from transformers import *
import json
import csv
import tempfile
import gensim

def lan_models():
    '''
    Le premier éteape s'est fait ici afin d'essayer 3 différents modèles de langage. Word2vec est effectué sur les textes importés.
    Le type de dossier de input et output n'est pas encore sur.
    Un probleme est le téléchargement embêtant des package chaque fois quand on exécute le modèle XLNet.
    :return: un dictionnaire qui contient les vecteurs de tokens sous différents LM
    '''
    models = [(OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
              (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
              (XLNetModel, XLNetTokenizer, 'xlnet-base-cased')]
    last_hidden_states = {'openai-gpt': [], 'transfo-xl-wt103': [], 'xlnet-base-cased': []}

    for model_class, tokenizer_class, pretrained_weight in models:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        model = model_class.from_pretrained(pretrained_weight)

        # with open('./contexte.csv') as csv_file:
        #     contexte = csv.reader(csv_file, delimiter='\n')
        #     for paragraphe in contexte:
        #         input_ids = torch.tensor([tokenizer.encode(paragraphe, add_special_tokens=True)])
        #         with torch.no_grad():
        #             last_hidden_states.append(model(input_ids)[0])

        with open('./contexte.txt','r') as txt_file:
            line = txt_file.readlines()
            for paragraphe in line:
                input_ids = torch.tensor([tokenizer.encode(paragraphe, add_special_tokens=True)])
                with torch.no_grad():
                    last_hidden_states[pretrained_weight].append(model(input_ids)[0])
                    print(line, '>>>to vector>>>', model(input_ids)[0], '\n')

    return last_hidden_states
        # with open('./contexte_encodage_'+pretrained_weight+'.json', "w") as json_f:
        #     for vector in last_hidden_states:
        #         json.dump(vector, json_f)

def model():
    '''
    une fonction similaire à lan_models() mais en utilisant qu'un MdL.
    :return: une liste qui contient tous les vecteurs générés en fonction des tokens.
    '''
    models = [(XLNetModel, XLNetTokenizer, 'xlnet-base-cased')]
    last_hidden_states = []
    for model_class, tokenizer_class, pretrained_weight in models:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        model = model_class.from_pretrained(pretrained_weight)


        with open('./contexte.txt','r') as txt_file:
            line = txt_file.readlines()
            for paragraphe in line:
                input_ids = torch.tensor([tokenizer.encode(paragraphe, add_special_tokens=False)])
                with torch.no_grad():
                    last_hidden_states.append(model(input_ids)[0])
                    print(line, '>>>to vector>>>', model(input_ids)[0], '\n')

    return last_hidden_states


if __name__ == '__main__':
    # model()
    model_path = 'frwiki.gensim'
    w2v_model = gensim.models.Word2Vec.load(model_path)

    # torch_roi = torch.FloatTensor(w2v_model.wv['roi'])
    # torch_ordinateur = torch.FloatTensor(w2v_model.wv['ordinateur'])
    # print(torch.dot(torch_roi,torch_ordinateur))
    # for i, word in enumerate(w2v_model.wv.vocab):
    #     if i == 10:
    #         break
    #     print(word)
    try:
        print(w2v_model.wv.similarity('pomme','pommes'))
    except KeyError:
        print("no this word")