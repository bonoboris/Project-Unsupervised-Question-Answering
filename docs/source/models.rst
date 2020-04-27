Models
======

Natural question / answers pairs generation full process depends on 2 NLP models:

- For `NER` / `Named Entity Recognition` it relies on `SpaCy <https://spacy.io/>`_ french model
- For `Constituency Parsing` it relies on `Benepar <https://pypi.org/project/benepar/>`_ french constituency parser

Installation
------------

Ensure your the virtual environnement you're using is activated.

Both model can be installed with the command:::

    uqa download all

The command will install:

* SpaCy ``fr_core_news_md`` model, the biggest of the 2 `pre-trained french available models <https://spacy.io/models/fr>`_
* Benepar ``benepar_fr`` model, the single `pre-trained french model available <https://github.com/nikitakit/self-attentive-parser#available-models>`_.
