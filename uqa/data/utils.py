import os
from os import path
import json
import random as rd
import pickle
from encodings import utf_8
from itertools import chain

DATA_PATH = path.dirname(__file__)

def json_discover(dirpath):
    """Explore folder `dirpath` yield any *.json file path.

    Args
    ----
        dirpath: Path-like
            the folder to explore
    Yields
    ------
        path: str
            the path of json from which `json_content` is read.
    """
    for subdirpath, _, files in os.walk(dirpath):
        for filename in files:
            if path.splitext(filename)[1] == ".json":
                yield path.join(subdirpath, filename)


def json_opener(path_it):
    """Open and yield json file from `path_it`.

    Args
    ----
        path_it: Iterable[Path-like]
            the files paths to open  
    Yields
    ------
        path: str
            the path of json from which `json_content` is read.
        json_content: dictionary
    """
    for filepath in path_it:
        doc = None
        with open(filepath, 'r') as file:
            doc = json.load(file)
        if doc:
            yield filepath, doc
        else:
            print(f"Unable to read {filepath}")


def json_loader(dirpath):
    """Explore folder `dirpath` yield any *.json file path and content.

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
    yield from json_opener(json_discover(dirpath))


def pickle_loader(fpath):
    """Load and yield a single file pickled dataset. """
    with open(fpath, 'rb') as file:
        content = pickle.load(file, encoding='utf8')
    if content:
        yield fpath, content


def pickle_dumper(jsonlike_it):
    for fpath, content in jsonlike_it:
        with open(fpath, 'wb') as file:
            pickle.dump(content, file)
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


def change_last_dir(input_path:str, new_dirname:str):
    dirs = input_path.split("/")
    if dirs[-1] == "":
        return "/".join(chain(dirs[:-2], (new_dirname, "")))
    else:
        return "/".join(chain(dirs[:-1], (new_dirname,)))


def change_dir(docs_it, from_dir, to_dir):
    """Replace from_dir to to_dir in docs_it first elements."""
    for el in docs_it:
        if isinstance(el, str):
            yield el.replace(from_dir, to_dir)
        else:
            fpath, *others = el
            new_fpath = fpath.replace(from_dir, to_dir)
            yield (new_fpath,) + tuple(others)

def add_suffix(docs_it, suffix):
    """Add `suffix` to the file names before the extension."""
    for fpath, fcontent in docs_it:
        dirpath, filename = path.split(fpath)
        name, ext = path.splitext(filename)
        new_fpath = path.join(dirpath, f"{name}{suffix}{ext}")
        yield new_fpath, fcontent

def count_articles(jsonlike) -> int:
    """Count number of article in a jsonlike array, where each element is an article."""
    return len(jsonlike)


def count_contexts(json_articles_ds) -> int:
    """Count number of contexts in a jsonlike array, with standard structure."""
    return sum((len(article["contexts"]) for article in json_articles_ds))


def clean(docs_it):
    import re
    xml_balise_ref = re.compile(r"<\w*>.+<\/\w+>")
    for fpath, fjson in docs_it:
        print("Processing", fpath)
        for article in fjson:
            for cont in article["contexts"]:
                matchs = xml_balise_ref.findall(cont["text"])
                if matchs:
                    print(f"file: {fpath}")
                    print(f"article: {article['title']} [{article['id_doc']}]")
                    print(f"context: {cont['id_context']}")
                    print(f"\t {matchs}")

if __name__ == "__main__":
    it = (("foo/01", 0, 1), ("foo/02", 0, 1), ("foo/03", 5, "baz"))
    it2 = ("foo/01", "foo/02", "foo/03")
    it3 = (("foo/01",), ("foo/02",), ("foo/03",))
    for el in change_dir(it2, "foo", "bar"):
        print(el)

