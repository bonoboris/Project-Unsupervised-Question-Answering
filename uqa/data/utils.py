from typing import Iterable
import os
from os import path
import ujson as json
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


def count_contexts(json_fcontent) -> int:
    """Count number of contexts in a jsonlike array, with standard structure."""
    return sum((len(article["contexts"]) for article in json_fcontent))

class JsonLoader():
    def __init__(self, dirpath: str, sort_by_filename=True, verbose=True):
        self.sort_by_filename = sort_by_filename
        self.verbose = verbose
        self._dirpath:str = dirpath
        self._num_files:int = -1
        self._it = json_opener(self.iter_filepath())
        self._it_cnt = 0
    
    def iter_filepath(self) -> Iterable[str]:
        it = json_discover(self.dirpath)
        if self.sort_by_filename:
            it = sorted(it)
        else:
            l = list(it)
            rd.shuffle(l)
            it = iter(l)
        return it

    @property
    def dirpath(self) -> int:
        return self._dirpath

    @property
    def num_files(self) -> int:
        if self._num_files == -1:
            self._num_files = sum(1 for _ in self.iter_filepath())
        return self._num_files

    def __iter__(self):
        return self
    
    def __next__(self):
        fpath, fcontent = next(self._it)
        self._it_cnt += 1
        if self.verbose:
            print(f"[{self._it_cnt} / {self.num_files}] Loaded {fpath}")
        return fpath, fcontent



def resplit(jsonfile_it, new_dirpath:str, filename_template:str, num_article_per_file:int) -> str:
    num = 0
    new_fcontent = []
    num_files = getattr(jsonfile_it, "num_files", "?")
    for i, (fpath, fcontent) in enumerate(jsonfile_it):
        print(f"Processing file [{i+1} / {num_files}]: {fpath}")
        for article in fcontent:
            if len(new_fcontent) == num_article_per_file:
                new_fpath = path.join(new_dirpath, filename_template.format(num))
                yield new_fpath, new_fcontent
                num += 1
                new_fcontent = list()
            new_fcontent.append(article)
    # Yield last if not empty
    if new_fcontent:
        new_fpath = path.join(new_dirpath, filename_template.format(num))
        yield new_fpath, new_fcontent

# def clean(docs_it):
#     import re
#     xml_balise_ref = re.compile(r"<\w*>.+<\/\w+>")
#     for fpath, fjson in docs_it:
#         print("Processing", fpath)
#         for article in fjson:
#             for cont in article["contexts"]:
#                 matchs = xml_balise_ref.findall(cont["text"])
#                 if matchs:
#                     print(f"file: {fpath}")
#                     print(f"article: {article['title']} [{article['id_doc']}]")
#                     print(f"context: {cont['id_context']}")
#                     print(f"\t {matchs}")

if __name__ == "__main__":
    def fix_context_id(json_file_it):
        for fpath, fcontent in json_file_it:
            for article in fcontent:
                for id_context, context in enumerate(article["contexts"]):
                    context["id_context"] = id_context
            yield fpath, fcontent
    
    data_path = path.join(DATA_PATH, "ga_const2")
    for fpath in json_dumper(fix_context_id(JsonLoader(data_path)), override=True):
        print(f"Saved {fpath}")
