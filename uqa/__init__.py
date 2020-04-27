""" Unsupervised question answer pairs generation in `French`.

Main featutes / processing pipeline:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Dataset cleaning
2. Named entity recognition [#spacy-fr]_
3. Constituency parsing [#benepar]_ [#benepar-paper1]_ [#benepar-paper2]_
4. Question / answer pairs generation

The implemented method is based on Kayo Yin (Illuin Technology)
`article <https://medium.com/illuin/unsupervised-question-answering-4758e5f2be9b>`_ [#uqa-medium].

Question / answer generation key points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using constituency parsing and named entity results, sub-sentences verifying the following constraints are matched:

- The sub-sentence is made of:

    1. a subject syntagm
    2. a stative verb
    3. a predicative nominal

- The subject syntagm contains one and only one named entity

The subject is used to genrated the answer while the verb and the predicative is used to generate the question.

Examples
--------
Process a single json file ``"data.json"`` in `default` format with CLI::

    $ uqa clean data.json clean.json
    $ uqa ner clean.json ner.json
    $ uqa constituency ner.json constituency.json
    $ uqa qas constituency.json uqa.json
    $ uqa show -df fquad fquad.json

Process a single json file ``"data.json"`` in `default` format with a single script::

    from uqa import dataset, clean, ner, constituency, qa_gen, fquad_utils

    if __name__ == "__main__"
        # Load data
        dataloader = dataset.FileDataLoader("data.json")
        # Process data
        processed_it = qa_gen.generate_qas_dl(constituency(ner(clean(dataloader))))
        # Convert data
        fquad_it = fquad_utils.default_to_fquad_dl(processed_it)
        # Save data
        path_modifier = dataset.DataDumper.path_replacer("fquad.json")
        dataset.DataDumper(path_modifier=path_modifier).save(fquad_it)

References
----------
.. [#uqa-medium] `Unsupervised Question Answering,
    Oct 2019 <https://medium.com/illuin/unsupervised-question-answering-4758e5f2be9b>`_
.. [#spacy-fr] `Spacy french model <https://spacy.io/models/fr>`_
.. [#benepar] `Berkley Neural Parser <https://github.com/nikitakit/self-attentive-parser>`_
.. [#benepar-paper1] `Constituency Parsing with a Self-Attentive Encoder,
    Nikita Kitaev, Dan Klein, May 2018,  <https://arxiv.org/pdf/1805.01052.pdf>`_
.. [#benepar-paper2] `Multilingual Constituency Parsing with Self-Attention and Pre-Training,
    Nikita Kitaev, Steven Cao, Dan Klein, Jun 2019,  <https://arxiv.org/abs/1812.11760.pdf>`_
"""
