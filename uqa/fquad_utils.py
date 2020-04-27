"""FQuAD data format related operations.

Implement conversion from `default` data format to `FQuAD` data format and vice-versa.
Conversions are not loss-less.

See Also
--------
:doc:`Data formats </data_formats>`
"""

import logging

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def fquad_to_default(fcontent: dataset.TJson, base_article_id: int = 0, include_qas: bool = False) -> dataset.TJson:
    """Convert a data container from `FQuAD` format to `default` format.

    Parameters
    ----------
    fcontent: :obj:`.TJson`
        Data in `FQuAD` format
    base_article_id: int, default=0
        ID to use for the first article, ID are generated increadingly.
    include_qas: bool, default=False
        If ``True`` convert ``qas`` fields, else discard it.

    Returns
    -------
    :obj:`.TJson`
        Data converted in `default` format
    """
    default_fcontent = list()
    for num_article, article in enumerate(fcontent["data"]):
        default_article = dict(title=article["title"], id_article=base_article_id + num_article, contexts=list())
        for num_context, para in enumerate(article["paragraphs"]):
            cont = dict(id_context=num_context, text=para["context"])
            if include_qas:
                default_qas = list()
                for qa_dct in para["qas"]:
                    ans_start = qa_dct["answers"][0]["answer_start"]
                    ans_end = ans_start + len(qa_dct["answers"][0]["text"])
                    default_ans = dict(start=ans_start, end=ans_end, label="Ans")
                    default_qa = dict(question=qa_dct["question"], answer=default_ans)
                    default_qas.append(default_qa)
                cont["qas"] = default_qas
            default_article["contexts"].append(cont)
        default_fcontent.append(default_article)
    return default_fcontent


def fquad_to_default_dl(data_it: dataset.DataIterable, include_qas=False) -> dataset.DataIterable:
    """Convert a dataset from `FQuAD` format to `default` format.

    Parameters
    ----------
    data_it: :obj:`.DataIterable`
        Dataset iterable (`FQuAD` format)
    include_qas: bool, default=False
        If ``True`` convert ``qas`` field, else discard it.

    Returns
    -------
    :obj:`.DataIterable`
        Dataset iterable (`default` format)
    """
    num_article = 0
    for fpath, fcontent in data_it:
        default_fcontent = fquad_to_default(fcontent, num_article, include_qas=include_qas)
        yield fpath, default_fcontent
        num_article += len(default_fcontent)


def default_to_fquad(fcontent: dataset.TJson, version: str = "0.1") -> dataset.TJson:
    """Convert a data container from `default` format to `FQuAD` format.

    Input data are expected to contain `qas` field.

    Parameters
    ----------
    fcontent: :obj:`.TJson`
        Data in `FQuAD` format
    version: str, default='0.1'
        version string to use.

    Returns
    -------
    :obj:`.TJson`
        Data converted in `FQuAD` format
    """
    qa_count = 0
    data = list()
    for article in fcontent:
        squad_article = dict(title=article["title"])
        paragraphs = list()
        for context in article["contexts"]:
            para = dict(context=context["text"], qas=list())
            for qa_dict in context["qas"]:
                question, answer_dict = qa_dict["question"], qa_dict["answer"]
                squad_qa = dict(question=question, id=qa_count)
                answer_start = answer_dict["start"]
                answer_text = context["text"][answer_start : answer_dict["end"]]
                squad_answer = [dict(text=answer_text, answer_start=answer_start)]
                squad_qa["answers"] = squad_answer
                para["qas"].append(squad_qa)
                qa_count += 1
            paragraphs.append(para)
        squad_article["paragraphs"] = paragraphs
        data.append(squad_article)
    return dict(version=version, data=data)


def default_to_fquad_dl(data_it: dataset.DataIterable, version="0.1") -> dataset.DataIterable:
    """Convert a dataset from `default` format to `FQuAD` format.

    Parameters
    ----------
    data_it: :obj:`.DataIterable`
        Dataset iterable (`default` format)
    version: str, default='0.1'
        version string to use.

    Returns
    -------
    :obj:`.DataIterable`
        Dataset iterable (`FQuAD` format)
    """
    for fpath, fcontent in data_it:
        yield fpath, default_to_fquad(fcontent, version)
