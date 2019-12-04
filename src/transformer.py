import torch
from transformers import *
import json
import csv
import tempfile


def lan_models():
    """
    The first step was done here to try 3 different language models. Word2vec is performed on imported texts.
  The input and output file type is not yet decided (json or csv).
  One problem is the downloading of same packages every time when running the XLNet model.
    :return:a dictionary that contains token vectors under different LMs
    """
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

        with open('./contexte.txt', 'r') as txt_file:
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
    """
        a function similar to lan_models () but using only one LM
    :return: a list that contains token vectors under the chose LM
    """
    models = [(OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt')]
    last_hidden_states = []
    for model_class, tokenizer_class, pretrained_weight in models:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        model = model_class.from_pretrained(pretrained_weight)

        with open('./contexte.txt', 'r') as txt_file:
            line = txt_file.readlines()
            for paragraphe in line:
                input_ids = torch.tensor([tokenizer.encode(paragraphe, add_special_tokens=False)])
                with torch.no_grad():
                    last_hidden_states.append(model(input_ids)[0])
                    print(line, '>>>to vector>>>', model(input_ids)[0], '\n')

    return last_hidden_states


if __name__ == '__main__':
    model()

