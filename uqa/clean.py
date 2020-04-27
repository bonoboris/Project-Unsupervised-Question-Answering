"""Cleaning and preprocessing tools.

Impletent a 'clean' feature working at character level and a 'filter' feature working
at context level.
"""

import logging
import string
import unicodedata
from collections import defaultdict
from typing import DefaultDict, List, Set

from uqa import dataset

#: Valid letters in French (ascii letters and accented letters)
ALPHA = list(string.ascii_letters)
ALPHA.extend("À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ à â ä ç é è ê ë î ï ô ö ù û ü ÿ æ œ Æ Œ".split(" "))

#: List of allowed characters
ALLOWED = list(ALPHA)
ALLOWED.extend("… § ‰".split(" "))

#: Character replacement rule based on categorie
REPLACE_CAT = {"Zs": " ", "Pd": "-"}

#: Character replacement table, override :obj:`REPLACE_CAT`
REPLACE_DICT = {
    "、": ",",
    "，": ",",
    "•": "-",
    "«": '"',
    "»": '"',
    "”": '"',
    "“": '"',
    "〟": '"',
    "„": '"',
    "〝": '"',
    "’": "'",
    "‛": "'",
    "‘": "'",
    "′": "'",
    "″": "''",
    "！": "!",
    "¡": "",
    "＆": "&",
    "·": ".",
    "・": ".",
    "（": "(",
    "）": ")",
    "【": "(",
    "】": ")",
    "〈": "(",
    "〉": ")",
    "†": " mort ",
    "：": ":",
    "։": ":",
    "‹": "<",
    "›": ">",
}


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def count_alpha(txt: str) -> int:
    """Return the number of letters in `txt`.

    Parameters
    ----------
    txt: str
        A string

    Returns
    -------
    int
        the number of character of `txt` that are in :obj:`ALPHA`
    """
    return len([c for c in txt if c in ALPHA])


def extract_bad_char(txt: str) -> Set[str]:
    """Return the set of non-allowed characters in `txt`.

    Parameters
    ----------
    txt: str
        The string to process.

    Returns
    -------
    set of str
        set of characters of `txt` that are not in :obj:`ALLOWED`
    """
    return set(txt).difference(ALLOWED)


def classify_bad_chars(bad_chars: Set[str]) -> DefaultDict[str, List[str]]:
    """Classify `bad_chars` set into unicode categories.

    Parameters
    ----------
    bad_chars: set of character str

    Returns
    -------
    defaultdict
        keys are unicode categories str, values are lists of character str

    See Also
    --------
    `List of unicode categories <http://www.unicode.org/reports/tr44/#General_Category_Values>`_
    """
    bad_chars_cat = defaultdict(list)
    for c in bad_chars:  # pylint: disable=invalid-name
        bad_chars_cat[unicodedata.category(c)].append(c)
    return bad_chars_cat


def clean_text(txt: str) -> str:
    """Clean and return `txt` either removing or replacing illegal characters.

    Illegal characters are characters not part of :const:`ALLOWED`.
    Illegal characters are replaced if they have an entry in :const:`REPLACE_DICT` or if their unicode category
    have an entry in :const:`REPLACE_CAT`.

    Parameters
    ----------
    txt: str
        The string to process

    Returns
    -------
    ret: str
        The string with illegal characters eiter replaced or removed.
    """
    bad_chars = extract_bad_char(txt)
    replace_dict = dict()
    for bad_char in bad_chars:
        if bad_char in REPLACE_DICT:
            replace_dict[bad_char] = REPLACE_DICT[bad_char]
        else:
            bad_char_cat = unicodedata.category(bad_char)
            if bad_char_cat in REPLACE_CAT:
                replace_dict[bad_char] = REPLACE_DICT[bad_char_cat]

    for bad_char in bad_chars:
        txt = txt.replace(bad_char, replace_dict.get(bad_char, ""))
    return txt


def clean(fcontent: dataset.TJson) -> dataset.TJson:
    """Clean `fcontent` json-like container in 'default' data format.

    Parameters
    ----------
    fcontent: :obj:`.TJson`
        Json-like container in `default` data format

    Returns
    -------
    cleaned_fcontent: :obj:`.TJson`
        The processed data in `default` data format
    """
    for article in fcontent:
        article["title"] = clean_text(article["title"])
        for context in article["contexts"]:
            context["text"] = clean_text(context["text"])
    return fcontent


def clean_dl(data_it: dataset.DataIterable):
    """Iterate through the dataset in 'default' data format.

    Parameters
    ----------
    data_it: DataIterable
        Dataset iterable, elements must be pairs (fpath, fcontent) where fpath is the data file path and
        fcontent is the file's content

    Yields
    ------
    fpath: str
        The processed file path
    filtered_fcontent: json-like
        The processed file content in default data format
    """
    for fpath, fcontent in data_it:
        print(f"Processing {fpath}")
        for article in fcontent:
            for context in article["contexts"]:
                context["text"] = clean_text(context["text"].encode().decode("utf8"))
        yield fpath, fcontent


def filter_contexts(fcontent: dataset.TJson, min_num_alpha: int = 10) -> dataset.TJson:
    """Filter contexts of all articles in `fcontent`.

    Parameters
    ----------
    fcontent: json-like
        Json-like container in default data format
    min_num_alpha: int
        Minimum number of letters if the context to be valid.

    Returns
    -------
    filtered_fcontent: json-like
        The processed data in default data format
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


def filter_contexts_dl(data_it: dataset.DataIterable, min_num_alpha: int = 10, detailed=False):
    """Filter all contexts in the dataset.

    Parameters
    ----------
    data_it: DataIterable
        Dataset iterable, elements must be pairs (fpath, fcontent) where fpath is the data file path and
        fcontent is the file's content
    min_num_alpha: int, optional
        Minimum number of letters if the context to be valid, defaults to 10.
    detailed: bool, optional
        If True, logs per file number of removed context

    Yields
    ------
    fpath: str
        The processed file path
    filtered_fcontent: json-like
        The processed file content in default data format
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
