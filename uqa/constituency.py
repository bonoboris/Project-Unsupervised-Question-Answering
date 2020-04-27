"""Constituency parsing with benepar using spacy framework.

Notes
-----
At the time of writing `benepar` latest release version (0.1.2) is not compatible with tensorflow 2., either
use a tensorflow 1. version or change `benepar.base_parser.py` first line from
```import tensorflow as tf```
to
```import tensorflow.compat.v1 as tf```
"""

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
    """Recursively convert a spacy Span object with parsed consituency information into a `uqa.context_utils.LabelNode`.

    Parameters
    ----------
    span: spacy.tokens.Span
        A spacy Span object populated with benepar component prediction in its `_` attribute
    Returns
    -------
    :class:`.context_utils.LabelNode`
        The constituent hierarchy as a ``LabelNode`` instance.
    """
    node = context_utils.LabelNode(span.start_char, span.end_char, "|".join(span._.labels))
    node.children = [span_to_node(child) for child in span._.children]
    return node


def constituency(fcontent: dataset.TJson, model: spacy.language.Model, detailed: bool = False) -> dataset.TJson:
    """Perform constituency parsing on a 'default' structure data json-like object.

    Use `Benepar` constituency parsing model but still relies on `SpaCy parser` stage outputs.

    Parameters
    ----------
    fcontent: :obj:`.TJson`
        A json-like data object with `default` structure.
    model: spacy.language.Language
        A loaded SpaCy model with `parser` and `benepar.spacy_pluggin.BeneparComponent` in the pipeline
    detailed: bool, default=False
        If ``True`` log per article progress

    Returns
    -------
    :obj:`.TJson`
        The processed data
    """
    for num_article, article in enumerate(fcontent):
        if detailed:
            logger.info(f"Processing article {num_article + 1} / {len(fcontent)}")
        for cont in article["contexts"]:
            spacy_doc = model(cont["text"])
            consts_json = [span_to_node(sent).to_json() for sent in spacy_doc.sents]
            cont["constituency"] = consts_json
    return fcontent


def constituency_dl(
    data_it: dataset.DataIterable, model_name: str = "fr_core_news_md", detailed: bool = False
) -> dataset.DataIterable:
    """Perform constituency parsing on a dataset.

    Use `Benepar` constituency parsing model but still relies on `SpaCy` `parser` stage outputs.

    Parameters
    ----------
    data_it: :obj:`.DataIterble`
        A dateset iterable in `default` format.
    model_name: str, default="fr_core_news_md"
        The name of the spacy model to load, the model has to be locally installed prior to be used.
    detailed: bool, default=False
        If ``True`` log per article progress

    Returns
    -------
    :obj:`.DataIterble`
        The processed dateset iterable.
    """
    logger.info("Loading spacy model for constituency parsing")
    model = spacy.load(model_name, disable=["tagger", "ner"])
    logger.info("Spacy model for constituency parsing loaded")
    logger.info("Adding benepar component")
    model.add_pipe(spacy_plugin.BeneparComponent("benepar_fr"))
    logger.info("Benepar component added to the pipe")
    for fpath, fcontent in data_it:
        logger.debug(f"Performing constituency parsing on {fpath}")
        try:
            yield fpath, constituency(fcontent, model, detailed=detailed)
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"while processing constituency parsing on {fpath}")
