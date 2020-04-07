from typing import List, Set, Dict, Text, DefaultDict, Iterable
from utils import pickle_loader, pickle_dumper, add_suffix, json_dumper
from string import printable, ascii_letters
from unicodedata import normalize
import unicodedata
from collections import defaultdict
from itertools import chain, tee
from os import path

from colorama import Fore

from utils import DATA_PATH


def red(txt):
    return Fore.RED + txt + Fore.RESET


def highlight(txt: Text, els: Iterable[Text]) -> Text:
    for el in els:
        txt = txt.replace(el, red(el))
    return txt

alpha = list(ascii_letters)
alpha.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ".split(" "))
allowed = list(printable)
allowed.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ … § ‰".split(" "))


def count_alpha(txt: Text) -> int:
    return len([c for c in txt if c in alpha])

def extract_bad_char(txt) -> Set[Text]:
    return set(txt).difference(allowed)

def classify_bad_chars(bad_chars: Set[Text]) -> DefaultDict[Text, List[Text]]:
    bad_chars_cat = defaultdict(list)
    for c in bad_chars:
        bad_chars_cat[unicodedata.category(c)].append(c)
    return bad_chars_cat


def clean_text(txt:Text) -> Text:
    bad_chars_cat = classify_bad_chars(extract_bad_char(txt))
    replace_dict = dict()
    cat_replace = {"Zs": " ", "Pd": "-"}
    for cat, rep in cat_replace.items():
        for c in bad_chars_cat[cat]:
            replace_dict[c] = rep
    for cat in bad_chars_cat:
        if cat not in cat_replace and not cat.startswith("P"):
            for c in bad_chars_cat[cat]:
                if c in replace_dict:
                    print('Overriding existing replacement rule !')
                replace_dict[c] = ""
    
    replace_dict["、"] = ","
    replace_dict["，"] = ","
    replace_dict["•"] = "-"
    replace_dict["«"] = '"'
    replace_dict["»"] = '"'
    replace_dict["”"] = '"'
    replace_dict["“"] = '"'
    replace_dict["〟"] = '"'
    replace_dict["„"] = '"'
    replace_dict["〝"] = '"'
    replace_dict["’"] = "'"
    replace_dict["‛"] = "'"
    replace_dict["‘"] = "'"
    replace_dict["′"] = "'"
    replace_dict["″"] = "''"
    replace_dict["！"] = "!"
    replace_dict["¡"] = ""
    replace_dict["＆"] = "&"
    replace_dict["·"] = "."
    replace_dict["・"] = "."
    replace_dict["（"] = "("
    replace_dict["）"] = ")"
    replace_dict["【"] = "("
    replace_dict["】"] = ")"
    replace_dict["〈"] = "("
    replace_dict["〉"] = ")"
    replace_dict["†"] = " mort "
    replace_dict["："] = ":"
    replace_dict["։"] = ":"
    replace_dict["‹"] = "<"
    replace_dict["›"] = ">"
    replace_dict["‿"] = ""
    replace_dict["׳"] = ""
    replace_dict["״"] = ""
    replace_dict["‡"] = ""
    replace_dict["‖"] = ""
    replace_dict["״"] = ""

    for bad, good in replace_dict.items():
        txt = txt.replace(bad, good)

    return txt

def clean(jsonfile_it):
    for fpath, fcontent in jsonfile_it:
        print(f"Processing {fpath}")
        for article in fcontent:
            for context in article["contexts"]:
                context["text"] = clean_text(context["text"].encode().decode("utf8"))
        yield fpath, fcontent

def all_bad_chars(jsonfile_it, show_bad=True) -> Set[Text]:
    all_bad_chars = set()
    for _, fcontent in jsonfile_it:
        for article in fcontent:
            for context in article["contexts"]:
                bad_chars = extract_bad_char(context["text"])
                all_bad_chars.update(bad_chars)
                if bad_chars and show_bad:
                    print(highlight(context["text"], bad_chars))
                    
    return all_bad_chars

def filter_context(jsonfile_it):
    """ """
    sum_removed = 0
    sum_total = 0
    for fpath, fcontent in jsonfile_it:
        print(f"Processing {fpath}")
        removed = 0
        total = 0
        new_fcontent = []
        for article in fcontent:
            new_article = dict(
                id_article=article["id_article"],
                title=article["title"],
            )
            new_id_context = 0
            new_contexts = list()
            total += len(article["contexts"])
            for context in article["contexts"]:
                if count_alpha(context["text"]) < 10:
                    removed += 1
                else:
                    new_context = dict(
                        id_context = new_id_context,
                        text = context["text"]
                    )
                    new_contexts.append(new_context)
            new_article["contexts"] = new_contexts
            new_fcontent.append(new_article)
        
        sum_removed += removed
        sum_total += total
        print(f"Removed {removed} context out of {total}")
        yield fpath, new_fcontent
    print(f"TOTAL --> Removed {sum_removed} context out of {sum_total}")


def split(jsonfile_it, data_subdir:str, filename_template:str, num_article_per_split:int=10):
    dir_path = path.join(DATA_PATH, data_subdir)
    num = 0
    new_fcontent = []
    for _, fcontent in jsonfile_it:
        for article in fcontent:
            if len(new_fcontent) == num_article_per_split:
                new_fpath = path.join(dir_path, filename_template.format(num))
                yield new_fpath, new_fcontent
                num += 1
                new_fcontent = list()
            new_fcontent.append(article)


if __name__ == "__main__":
    from utils import json_loader

    print("--- CLEAN ---")
    filepath = "uqa/data/good_articles.pickle"
    for fpath in pickle_dumper(add_suffix(clean(pickle_loader(filepath)), "_clean")):
        print("Saved", fpath)

    print("--- FILTER ---")
    filepath = "uqa/data/good_articles_clean.pickle"
    for fpath in pickle_dumper(add_suffix(filter_context(pickle_loader(filepath)), "_filter")):
        print("Saved", fpath)

    print("--- SPLIT ---")
    filepath = "uqa/data/good_articles_clean_filter.pickle"
    for fpath in json_dumper(split(pickle_loader(filepath), "ga_json", "ga_{:03d}.json"), override=True):
        print("Saved", fpath)
