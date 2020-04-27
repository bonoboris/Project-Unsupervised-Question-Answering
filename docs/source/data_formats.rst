Data Formats
============

**UQA** package use 2 JSON data formats in UTF-8 encoding.

Note that the package support multi-file dataset,
we'll describe to strucure of one file of the dataset.

.. _default-data-format:

``Default`` data format
-----------------------

The ``default`` data format refers to our own JSON structure for storing data.

The minimal stucture is the following:::

    [
        {  # article dictionary
            "id_article": int,
            "title": str,
            "contexts": [
                {
                    "id_context": int,
                    "text": str,
                }, ...
            ]
        }, ...
    ]

Description
^^^^^^^^^^^

* A JSON file in default format is a array of dictionaries, where each dictionary represent an article.
* ``id_article``: a unique positive integer across the dataset
* ``title``: the article's title
* ``contexts``: a list of dictionaries where each dictionary represent a context / paragraph
* ``id_context``: a unique positive integer across all the article contexts
* ``text``: the context / paragraph raw content

Additional field
^^^^^^^^^^^^^^^^
Results of different processing steps are stored directly into this structure.
All processing steps are at context granularity hence the ``context`` dictionaries can contains the additional fields:

* ``entities``: a **list** of dictionaries with the structure:::

    {
        "start": int,   # entity start index in ``text`` field
        "end": int,     # entity end index in ``text`` field
        "label": str,   # entity label
    }

* ``constituents``: a **list** of dictionaries with the structure:::

    {
        "start": int,   # entity start index in ``text`` field
        "end": int,     # entity end index in ``text`` field
        "label": str,   # entity label

        "children":     # nested repeating structure
        [
            {"start": int, "end": int, "label": str, "children": [...]},
            {"start": int, "end": int, "label": str, "children": [...]},
        ]
    }

* ``qas``: a **list** of dictionaries with the structure:::

    {
        question: str,
        answer: {
            "start": int,   # answer start index in ``text`` field
            "end": int,     # answer end index in ``text`` field
            "label": str,   # answer label
        }
    }

.. _fquad-data-format:

``FQuAD`` data format
---------------------
`FQuAD: French Question Answering Dataset <https://fquad.illuin.tech/>`_ is the dataset created and released by `Illuin Technology <https://www.illuin.tech/>`_ for training a
french question-answering model.

We use their format data for the output of the question answer generation step for inter-operability with the training script they provided.

Structure:::

    {
        "version": str
        "data": [
            {
              "title": str,
              "paragraphs": [
                    {
                        "context": str,  # raw text
                        "qas": [question_answer_dict, ...]
                    }, ...
                ]
            }, ...
        ]
    }

where each ``question_answer_dict`` have the following structure:::

    {
        "id": uid or int
        "question": str
        "answers":[
            {
                "answer_start": int,  # the start index in ``context``
                "text": str:  # always a part of context
            }, ...  # both FQuAD dataset and our generated dataset contains a single answer per question
        ]
    }

