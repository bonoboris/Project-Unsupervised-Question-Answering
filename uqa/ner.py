"""Named entity recognition with spacy."""

import logging
import spacy
from uqa.dataset import DataIterable, TJson

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def ner(fcontent: TJson, model: spacy.language.Model) -> TJson:
    """Perform NER on a 'default' structure data collection."""
    article = None
    context = None
    for article in fcontent:
        for context in article["contexts"]:
            doc = model(context["text"])
            context["entities"] = doc.to_json()["ents"]
    return fcontent


def ner_dl(data_it: DataIterable, model_name="fr_core_news_md") -> DataIterable:
    """Perform NER on a dataset."""
    logger.info("Loading spacy model for NER")
    model = spacy.load(model_name, disable=["tagger", "parser"])
    logger.info("Spacy model for NER loaded")
    for fpath, fcontent in data_it:
        logger.debug(f"Performing NER on {fpath}")
        yield fpath, ner(fcontent, model)
