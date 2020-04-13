"""fquad data format related functions."""

import logging

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def fquad_to_default(fcontent: dataset.TJson, base_article_id: int = 0, include_qas: bool = False) -> dataset.TJson:
    """Convert a data file's content from fquad format to 'default' format."""
    default_fcontent = list()
    for num_article, article in enumerate(fcontent["data"]):
        default_article = dict(title=article["title"], id_article=base_article_id + num_article, contexts=list())
        for num_context, para in enumerate(article["paragraphs"]):
            cont = dict(id_context=num_context, text=para["context"])
            if include_qas:
                raise NotImplementedError("Questions answer formating not implemented.")
            default_article["contexts"].append(cont)
        default_fcontent.append(default_article)
    return default_fcontent


def fquad_to_default_dl(data_it: dataset.DataIterable) -> dataset.DataIterable:
    """Convert a datset from fquad format to 'default' format."""
    num_article = 0
    for fpath, fcontent in data_it:
        default_fcontent = fquad_to_default(fcontent, num_article)
        yield fpath, default_fcontent
        num_article += len(default_fcontent)


def default_to_squad(fcontent: dataset.TJson, version: str = "0.1") -> dataset.TJson:
    """Convert a data file's content from fquad format to 'default' format."""
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
                squad_qa["answer"] = squad_answer
                para["qas"].append(squad_qa)
                qa_count += 1
            paragraphs.append(para)
        squad_article["paragraphs"] = paragraphs
        data.append(squad_article)
    return dict(version=version, data=data)


def default_to_squad_dl(data_it: dataset.DataIterable, version="0.1") -> dataset.DataIterable:
    """Convert a datset from default format to squad format. Discard ner & constituency fields."""
    for fpath, fcontent in data_it:
        yield fpath, default_to_squad(fcontent, version)
