"""Cleaning and preprocessing tools."""

import logging
import string
import unicodedata
from collections import defaultdict
from typing import DefaultDict, List, Set, Text

from uqa.dataset import DataIterable, TJson

ALPHA = list(string.ascii_letters)
ALPHA.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ".split(" "))
ALLOWED = list(string.printable)
ALLOWED.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ … § ‰".split(" "))

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def count_alpha(txt: Text) -> int:
    """Count the number of letters in `txt`."""
    return len([c for c in txt if c in ALPHA])


def extract_bad_char(txt) -> Set[Text]:
    """Return the set of non-allowed characters in `txt`."""
    return set(txt).difference(ALLOWED)


def classify_bad_chars(bad_chars: Set[Text]) -> DefaultDict[Text, List[Text]]:
    """Classify `bad_chars` set into unicode categories."""
    bad_chars_cat = defaultdict(list)
    for c in bad_chars:  # pylint: disable=invalid-name
        bad_chars_cat[unicodedata.category(c)].append(c)
    return bad_chars_cat


def clean_text(txt: Text) -> Text:
    """Clean `txt` either removing or replacing non allowed characters."""
    bad_chars_cat = classify_bad_chars(extract_bad_char(txt))
    replace_dict = dict()
    cat_replace = {"Zs": " ", "Pd": "-"}
    for cat, rep in cat_replace.items():
        for c in bad_chars_cat[cat]:  # pylint: disable=invalid-name
            replace_dict[c] = rep
    for cat in bad_chars_cat:
        if cat not in cat_replace and not cat.startswith("P"):
            for c in bad_chars_cat[cat]:  # pylint: disable=invalid-name
                if c in replace_dict:
                    print("Overriding existing replacement rule !")
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


def clean(fcontent: TJson) -> TJson:
    """Clean `fcontent` jsonlike collection with default structure."""
    for article in fcontent:
        article["title"] = clean_text(article["title"])
        for context in article["contexts"]:
            context["text"] = clean_text(context["text"])
    return fcontent


def clean_dl(data_it: DataIterable):
    """Iterate through the dataset and clean the contents."""
    for fpath, fcontent in data_it:
        print(f"Processing {fpath}")
        for article in fcontent:
            for context in article["contexts"]:
                context["text"] = clean_text(context["text"].encode().decode("utf8"))
        yield fpath, fcontent


# def all_bad_chars(jsonfile_it, show_bad=True) -> Set[Text]:
#     all_bad_chars = set()
#     for _, fcontent in jsonfile_it:
#         for article in fcontent:
#             for context in article["contexts"]:
#                 bad_chars = extract_bad_char(context["text"])
#                 all_bad_chars.update(bad_chars)
#                 if bad_chars and show_bad:
#                     print(highlight(context["text"], bad_chars))

#     return all_bad_chars


# def red(txt):
#     return Fore.RED + txt + Fore.RESET


# def highlight(txt: Text, els: Iterable[Text]) -> Text:
#     for el in els:
#         txt = txt.replace(el, red(el))
#     return txt


def filter_contexts(fcontent: TJson, min_num_alpha: int = 10) -> TJson:
    """Filter contexts of all articles in `fcontent`.

    Args
    ----
        min_num_alpha: int
            Minimum number of letters if the context to be valid.
    """
    num_removed = 0
    num_total = 0
    new_fcontent = []
    for article in fcontent:
        new_article = dict(id_article=article["id_article"], title=article["title"],)
        new_id_context = 0
        new_contexts = list()
        num_total += len(article["contexts"])
        for context in article["contexts"]:
            if count_alpha(context["text"]) < min_num_alpha:
                num_removed += 1
            else:
                new_context = dict(id_context=new_id_context, text=context["text"])
                new_contexts.append(new_context)
                new_id_context += 1
        new_article["contexts"] = new_contexts
        new_fcontent.append(new_article)
    return new_fcontent, (num_removed, num_total)


def filter_contexts_dl(data_it: DataIterable, min_num_alpha: int = 10, detailed=False):
    """Filter all contexts in the dataset.

    Args
    ----
        min_num_alpha: int
            Minimum number of letters if the context to be valid.
        detailed: bool
            Log per file number of context removed
    """
    sum_removed = 0
    sum_total = 0
    for fpath, fcontent in data_it:
        filtered, (num_removed, num_total) = filter_contexts(fcontent, min_num_alpha)
        if detailed:
            logger.info(f"Removed contexts: {num_removed} /  {num_total}")
        sum_removed += num_removed
        sum_total += num_total
        yield fpath, filtered
    logger.info(f"TOTAL: Removed contexts: {sum_removed} / {sum_total}")


# def split(jsonfile_it, data_subdir: str, filename_template: str, num_article_per_split: int = 10):
#     dir_path = path.join(DATA_PATH, data_subdir)
#     num = 0
#     new_fcontent = []
#     for _, fcontent in jsonfile_it:
#         for article in fcontent:
#             if len(new_fcontent) == num_article_per_split:
#                 new_fpath = path.join(dir_path, filename_template.format(num))
#                 yield new_fpath, new_fcontent
#                 num += 1
#                 new_fcontent = list()
#             new_fcontent.append(article)
#     # Yield last if not empty
#     if new_fcontent:
#         new_fpath = path.join(dir_path, filename_template.format(num))
#         yield new_fpath, new_fcontent
