import json
import os
from collections import OrderedDict


def read_lines_from_file(file_path, sep='\n'):
    path = os.path.abspath(file_path)
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"No file found with path: {path}")
    lines = list()
    with open(file_path, "r", encoding="utf8") as file:
        lines = [line for line in file.read().split(sep) if line != ""]
    return lines


def make_triplet_dict(title, contexts):
    document_dict = OrderedDict()
    document_dict["title"] = title
    triplets = [{"context": context, "question": None, "answer_span": None} for context in contexts]
    document_dict["triplets"] = triplets
    return document_dict


def save_json(dumpable, file_path):
    with open(file_path, "w") as file:
        file.write(json.dumps(dumpable, indent=4, ensure_ascii=False))


def read_json(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        return json.load(file, encoding="utf8")


def transform_txt_in_json_triplet(file_path):
    path = os.path.abspath(file_path)
    dirpath, filename = os.path.split(path)
    title = filename.split('.')[0]

    lines = read_lines_from_file(path)
    document_dict = make_triplet_dict(title, lines)

    target_file_path = os.path.join(dirpath, (title + ".json"))
    save_json(document_dict, target_file_path)


def main():
    txt_file_path = "../ressources/Jacques_Chirac_intro_manual.txt"
    json_file_path = "../ressources/Jacques_Chirac_intro_manual.json"
    print(read_json(json_file_path))


if __name__ == '__main__':
    main()
