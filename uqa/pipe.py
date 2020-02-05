from os import path, makedirs
from reading_wiki_dumps import wiki_extractor_parser
from ner import ner_gen
from data import DATA_PATH
import json


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
