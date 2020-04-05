from typing import List, Set, Dict, Text, DefaultDict, Iterable
from utils import pickle_loader, pickle_dumper, add_suffix
from string import printable
from unicodedata import normalize
import unicodedata
from collections import defaultdict
from itertools import chain, tee

from colorama import Fore

def red(txt):
    return Fore.RED + txt + Fore.RESET


def highlight(txt: Text, els: Iterable[Text]) -> Text:
    for el in els:
        txt = txt.replace(el, red(el))
    return txt

allowed = list(printable)
allowed.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ … § ‰".split(" "))


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
    cat_replace = {"Ws": " ", "Pd": "-"}
    for cat, rep in cat_replace.items():
        for c in bad_chars_cat[cat]:
            replace_dict[c] = rep
    for cat in bad_chars_cat:
        if cat not in cat_replace and not cat.startswith("P"):
            for c in bad_chars_cat[cat]:
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
        for article in fcontent:
            for context in article["contexts"]:
                context["text"] = clean_text(context["text"])
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

if __name__ == "__main__":
    for fpath in pickle_dumper(add_suffix(clean(pickle_loader("uqa/data/good_articles.pickle")), "_clean")):
        print("Done", fpath)