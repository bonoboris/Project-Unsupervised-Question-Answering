"""Read and parse wiki tar.bz2 dumps archives.

Notes
-----
This module is not properly integrated with the CLi and the package in general.
"""

import bz2
import os
import re
import logging

try:
    import ujson as json
except ModuleNotFoundError:
    import json

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def wiki_extractor_parser(dir_path: str) -> dataset.DataIterable:
    """Explore directory `dir_path` open archives and yield `default` formated data.

    Arguments
    ---------
    dir_path: str
        The path to the directory to explore. (all files are considered as archives)

    Yields
    ------
    str
        The extracted data file path
    :obj:`.dataset.TJson`
        The formated data in `default` data format.
    """
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            path_file = os.path.join(subdir, file)
            with bz2.open(path_file, "rb") as file:
                list_json_documents = file.readlines()
                file_json = list()
                for string_json in list_json_documents:
                    # Getting the Parsed Json
                    string_json_decoded = string_json.decode(encoding="utf-8")
                    wiki_article_json = json.loads(string_json_decoded)
                    # Parsing the Text into Paragraph
                    article_text = wiki_article_json["text"]
                    article_text = re.sub(r"\n+", "\n", article_text)
                    contexts = article_text.split("\n")
                    # Remove last and first element
                    contexts.pop(0)
                    contexts.pop(-1)
                    final_contexts = []
                    for id_context, context in enumerate(contexts):
                        final_contexts.append({"id_context": id_context, "text": context})
                    final_json_list = {
                        "id_article": int(wiki_article_json["id"]),
                        "title": wiki_article_json["title"],
                        "contexts": final_contexts,
                    }
                    file_json.append(final_json_list)
                yield path_file, file_json
