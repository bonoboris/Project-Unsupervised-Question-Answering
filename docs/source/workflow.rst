Basic workflow
==============

The basic worklow uses the CLI to perform 4 steps:

1. Corpus cleaning
2. NER
3. Constituency Parsing
4. Question / answers pairs generation

Assuming a JSON text corpus in :ref:`default data format <default-data-format>` located at ``${WORKDIR}/corpus.json``.

0. First activate your virtualenvironnemnt and go the your ``${WORKDIR}``:::

    source ${UQA_VIRTUALENV_FOLDER}/bin/activate
    cd ${WORKDIR}

1. Then perform cleaning:::

    uqa clean corpus.json clean.json

2. Then perform NER:::

    uqa ner clean.json ner.json

3. Then perform Constituency Parsing:::

    uqa constituency ner.json constituency.json

4. Finally perform question / answer pairs generation:::

    uqa qas constituency.json qas.json

The last created file ``qas.json`` is written in :ref:`fquad-data-format`.

Troubleshooting
---------------

Each call to ``uqa`` command generate logs in the file ``uqa.log`` (by default) in the directory from which the command is executed.

If NER or Constituency Parsing fails, first ensure the models are correctly installed (:doc:`models`).

| Benepar constituency parsing model is limited to sentences with 298 or less tokens.
| If the constituency parsing step raise the error ``ValueError: Sentence of length 299 is too long to be parsed``
  you might want to pin point which sentence is provoking it.
| The problematic sentence could be a really long sentence, some dirty data in the corpus or a badly parsed group of sentence.
| The 298 word limit is intrinsic to the model architecture and cannot be lifted; either remove or modify the problematic section in the corpus.

Notes
-----

* Constituency parsing step increase dataset size on disk by a factor of ~10.

* Constituency parsing is the slowest operation and can be really time consuming for big dataset,
  thus it is recommended to split the dataset in multiple files, doing so if an error occurs while processing a given file,
  only this file output will be lost.
