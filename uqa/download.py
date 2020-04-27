"""Download model and ressources and install them in their default location.

Downloadable ressources includes:

- SpaCy French model: ``fr_core_news_md`` (recommended) or ``fr_core_news_sm``
- Benepar French Constituency parser

Notes
-----
* SpaCy models are installed as python packages in `site-packages` directory
* Benepar models are downloaded with ``nltk.downlader`` and installed in NLTK default download directory
  (see `nltk documentation <www.nltk.org/api/nltk.html#nltk.downloader.Downloader.default_download_dir>`_)
"""

import logging

import benepar
import spacy


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def download_spacy_model(name: str = "fr_core_news_md") -> None:
    """Download SpaCy model `name`."""
    spacy.cli.download(name)


def download_benepar_model(name: str = "benepar_fr") -> None:
    """Download Benepar model `name`."""
    benepar.download(name)
