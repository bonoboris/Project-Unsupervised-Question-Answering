"""Named entity recognition with SpaCy french model."""

import logging

import spacy

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def ner(fcontent: dataset.TJson, model: spacy.language.Language) -> dataset.TJson:
    """Use `model` to perform NER on the 'default' structure data container `fcontent`.

    Parameters
    ----------
    fcontent: :obj:`.TJson`
        A json-like data object with `default` structure.
    model: spacy.language.Language
        A loaded SpaCy model with 'ner' pipe in the pipeline

    Returns
    -------
    :obj:`.TJson`
        The processed data
    """
    article = None
    context = None
    for article in fcontent:
        for context in article["contexts"]:
            doc = model(context["text"])
            context["entities"] = doc.to_json()["ents"]
    return fcontent


def ner_dl(data_it: dataset.DataIterable, model_name="fr_core_news_md") -> dataset.DataIterable:
    """Load spacy `model_name` perform NER on the 'default' structure dataset iterable `data_it`.

    Parameters
    ----------
    data_it: :obj:`.DataIterble`
        A dateset iterable in `default` format.
    model_name: str, default="fr_core_news_md"
        The name of the spacy model to load, the model has to be locally installed prior to be used.

    Returns
    -------
    :obj:`.DataIterble`
        The processed dateset iterable.
    """
    logger.info("Loading spacy model for NER")
    model = spacy.load(model_name, disable=["tagger", "parser"])
    logger.info("Spacy model for NER loaded")
    for fpath, fcontent in data_it:
        logger.debug(f"Performing NER on {fpath}")
        try:
            yield fpath, ner(fcontent, model)
        except:
            logger.exception(f"while performing NER on {fpath}:")
