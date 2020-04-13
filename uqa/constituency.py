"""Constituency parsing with benepar using spacy framework."""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# pylint: disable=wrong-import-position
from benepar import spacy_plugin
import spacy
import tensorflow as tf

from uqa import context_utils, dataset

# pylint: enable=wrong-import-position


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# pylint: disable=useless-suppression
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # pylint: disable=no-member
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)  # pylint: disable=no-member
# pylint: enable=useless-suppression


def span_to_node(span: spacy.tokens.Span) -> context_utils.LabelNode:
    """Recursively convert a spacy Span object with parsed consituency information into a context.LabelNode."""
    node = context_utils.LabelNode(span.start_char, span.end_char, "|".join(span._.labels))
    node.children = [span_to_node(child) for child in span._.children]
    return node


# def constituency_context(context_it: Iterable[Context]):
#     Model.model.add_pipe(BeneparComponent("benepar_fr"))
#     for context in context_it:
#         spacy_doc = Model.model(context.text, disable=["ner", "tagger"])
#         consts = list()
#         for sent in spacy_doc.sents:
#             consts.append(span_to_node(sent))
#         context.constituents = consts
#         yield context


def constituency(fcontent: dataset.TJson, model: spacy.language.Model) -> dataset.TJson:
    """Perform constituency parsing on a 'default' structure data collection."""
    for article in fcontent:
        print(f"Processing article {article['id_article']}")
        for cont in article["contexts"]:
            spacy_doc = model(cont["text"])
            consts_json = [span_to_node(sent).to_json() for sent in spacy_doc.sents]
            cont["constituency"] = consts_json
    return fcontent


def constituency_dl(data_it: dataset.DataIterable, model_name: str = "fr_core_news_md") -> dataset.DataIterable:
    """Perform constituency parsing on a dataset."""
    logger.info("Loading spacy model for constituency parsing")
    model = spacy.load(model_name, disable=["tagger", "ner"])
    logger.info("Spacy model for constituency parsing loaded")
    logger.info("Adding benepar component")
    model.add_pipe(spacy_plugin.BeneparComponent("benepar_fr"))
    logger.info("Benepar component added to the pipe")
    for fpath, fcontent in data_it:
        try:
            yield fpath, constituency(fcontent, model)
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Error while processing {fpath}")
