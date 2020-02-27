import itertools
import json
from os import path, makedirs

from uqa.spacywrapper import SpacyFrenchModelWrapper

from uqa.data import DATA_PATH
from uqa.reading_wiki_dumps import wiki_extractor_parser

test_data = [
    [
        {
            "id_doc": 0,
            "title": "FOOOOO",
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 0."
                },
                {
                    "id_context": 1,
                    "text": "Je suis à Paris avec mon ami Jacques."
                },
            ]
        },
        {
            "id_doc": 1,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 1."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 2."
                },
            ]
        }
    ],
    [
        {
            "id_doc": 2,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 3."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 4."
                },
            ]
        },
        {
            "id_doc": 3,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 5."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 6."
                },
            ]
        }
    ],
    [
        {
            "id_doc": 3,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 7."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 8."
                },
            ]
        },
        {
            "id_doc": 4,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 9."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 10."
                },
            ]
        }
    ],
]


def ner_gen(json_file_it):
    json_file_it, json_file_it_copy = itertools.tee(json_file_it)
    context_it = (context["text"] for _, json_file in json_file_it_copy for article in json_file for context in article["contexts"])
    docs_it = SpacyFrenchModelWrapper.model.pipe(context_it, disable=["parser", "tagger"])
    for file_path, json_file in json_file_it:
        for json_article in json_file:
            for json_context in json_article["contexts"]:
                doc = next(docs_it)
                json_context["entities"] = doc.to_json()["ents"]
        yield file_path, json_file


NER_DATA_FOLDER = "ner_json"


def generate_ner_json():
    prefix = path.join(DATA_PATH, "ner_json")
    for filepath, ner_json in ner_gen(wiki_extractor_parser()):
        print(f"{filepath} done...", end=" ")
        subdirs, filename = path.split(filepath)
        filename = ".".join(filename.split(".")[:-1]) + ".json"
        dstpath = path.join(prefix, subdirs, filename)
        if not path.exists(path.dirname(dstpath)):
            makedirs(path.dirname(dstpath))
        with open(dstpath, "w") as file:
            json.dump(ner_json, file, indent=None)
            print("saved !")


if __name__ == '__main__':
    generate_ner_json()
