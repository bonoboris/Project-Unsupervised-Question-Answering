import os
from os import path
import json
import random as rd
import pickle
from encodings import utf_8

DATA_PATH = path.dirname(__file__)


def json_loader(dirpath):
    """Explore folder `dirpath` yield any *.json file content and path.

    Args
    ----
        dirpath: Path-like
            the folder to explore
    Yields
    ------
        path: str
            the path of json from which `json_content` is read.
        json_content: dictionary
    """
    for subdirpath, _, files in os.walk(dirpath):
        for filename in files:
            if path.splitext(filename)[1] == ".json":
                filepath = path.join(subdirpath, filename)
                doc = None
                with open(filepath, 'r') as file:
                    doc = json.load(file)
                if doc:
                    yield filepath, doc


def pickle_loader(fpath):
    """Load and yield a single file pickled dataset. """
    with open(fpath, 'rb') as file:
        content = pickle.load(file, encoding='utf8')
    if content:
        yield fpath, content


def pickle_dumper(jsonlike_it):
    for fpath, content in jsonlike_it:
        with open(fpath, 'wb') as file:
            pickle.dump(content, file, encoding='utf8')
        yield fpath


def find_json_files(dirpath):
    for subdirpath, _, files in os.walk(dirpath):
        for filename in files:
            if path.splitext(filename)[1] == ".json":
                filepath = path.join(subdirpath, filename)
                yield filepath

def count_json_files(dirpath):
    return len(list(find_json_files(dirpath)))


def json_loader_shuffle(dirpath):
    """Explore folder `dirpath` yield any *.json file content and path in a random order.

    Args
    ----
        dirpath: Path-like
            the folder to explore
    Yields
    ------
        json_content: dictionary
        path: str
            the path of json from which `json_content` is read.
    """
    paths = list()
    for subdirpath, _, files in os.walk(dirpath):
        for filename in files:
            if path.splitext(filename)[1] == ".json":
                filepath = path.join(subdirpath, filename)
                paths.append(filepath)
    rd.shuffle(paths)
    for filepath in paths:
        with open(filepath, 'r') as file:
            doc = json.load(file)
        if doc:
            yield filepath, doc


def json_dumper(docs_it, override=False):
    """Dump json document iterator.

    Args
    ----
        docs_it: iterator[Json-like, path]
    Yields
    ------
        path of the saved file
    """
    for fpath, doc in docs_it:
        subdirpath = path.dirname(fpath)
        if not path.exists(subdirpath):
            os.makedirs(subdirpath)
        if not override and path.exists(fpath):
            raise FileExistsError(f"File {fpath} already exist, set argument `override` to True to override existing files.")
        with open(fpath, 'w', encoding='utf8') as file:
            json.dump(doc, file, ensure_ascii=False)
            yield fpath


def change_dir(docs_it, from_dir, to_dir):
    """Replace from_dir to to_dir in docs_it first elements."""
    for fpath, doc in docs_it:
        new_fpath = fpath.replace(from_dir, to_dir)
        yield new_fpath, doc


def count_articles(jsonlike) -> int:
    """Count number of article in a jsonlike array, where each element is an article."""
    return len(jsonlike)


def count_contexts(json_articles_ds) -> int:
    """Count number of contexts in a jsonlike array, with standard structure."""
    return sum((len(article["contexts"]) for article in json_articles_ds))


def clean(docs_it):
    import re
    xml_balise_ref = re.compile(r"<\w*>.+<\/\w+>")
    for fjson, fpath in docs_it:
        for article in fjson:
            for cont in article["contexts"]:
                matchs = xml_balise_ref.findall(cont["text"])
                if matchs:
                    print(f"file: {fpath}")
                    print(f"article: {article['title']} [{article['id_doc']}]")
                    print(f"context: {cont['id_context']}")
                    print(f"\t {matchs}")
