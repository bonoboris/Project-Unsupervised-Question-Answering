"""Validate 'FQuAD' format dataset.

Check for required fields existence, `answers` / `context` field accordance and `id` unicity accross a file.
"""
import logging

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def validate(fcontent: dataset.TJson) -> None:
    """Validate `fcontent` json-like data in `FQuAD` format.

    Check for required fields existence, `answers` / `context` field accordance and `id` unicity accross a file.
    """
    qa_ids = set()
    if "version" not in fcontent:
        raise ValueError("invalid FQuAD: missing `version` field at root level")
    if "data" not in fcontent:
        raise ValueError("invalid FQuAD: missing `data` field at root level")
    for idx_article, article in enumerate(fcontent["data"]):
        if "title" not in article:
            raise ValueError(f"invalid FQuAD: in `data[{idx_article}]`: missing `title` field")
        elif "paragraphs" not in article:
            raise ValueError(f"invalid FQuAD: in `data[{idx_article}]`: missing `paragraphs` field")
        for idx_paragraph, paragraph in enumerate(article["paragraphs"]):
            if "context" not in paragraph:
                raise ValueError(
                    f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}]`:" f"missing `context` field"
                )
            elif "qas" not in paragraph:
                raise ValueError(
                    f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}]`: missing `qas` field"
                )
            for idx_qa, qa in enumerate(paragraph["qas"]):  # pylint: disable=invalid-name
                if "id" not in qa:
                    raise ValueError(
                        f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}]`:"
                        f"missing `id` field"
                    )
                if qa["id"] in qa_ids:
                    raise ValueError(
                        f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}]`:"
                        f"`id` field value: {qa['id']} already in use"
                    )
                else:
                    qa_ids.add(qa["id"])
                if "question" not in qa:
                    raise ValueError(
                        f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}]`:"
                        f"missing `question` field"
                    )
                if "answers" not in qa:
                    raise ValueError(
                        f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}]`:"
                        f"missing `answers` field"
                    )
                if len(qa["answers"]) == 0:
                    raise ValueError(
                        f"invalid FQuAD: in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}]`:"
                        f"`answers` field must contain at least 1 answer"
                    )
                for idx_answer, answer in enumerate(qa["answers"]):
                    if "answer_start" not in answer:
                        raise ValueError(
                            f"invalid FQuAD:"
                            f"in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}].answers[{idx_answer}]`:"
                            f"missing `answer_start` field"
                        )
                    if "text" not in answer:
                        raise ValueError(
                            f"invalid FQuAD:"
                            f"in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}].answers[{idx_answer}]`:"
                            f"missing `text` field"
                        )
                    ans_start = answer["answer_start"]
                    ans_text = answer["text"]
                    context = paragraph["context"]
                    if (
                        ans_start < 0
                        or ans_start >= len(context) - len(ans_text)
                        or context[ans_start : ans_start + len(ans_text)] != ans_text
                    ):
                        raise ValueError(
                            f"invalid FQuAD:"
                            f"in`data[{idx_article}].paragraphs[{idx_paragraph}].qas[{idx_qa}].answers[{idx_answer}]`:"
                            f"values for `answer_start` and `text` conflict with `context` field value."
                        )


def validate_dl(data_it: dataset.DataIterable) -> None:
    """Validate a dataset in `FQuAD` format.

    Check for required fields existence, `answers` / `context` field accordance and `id` unicity accross a file.
    """
    for fpath, fcontent in data_it:
        validate(fcontent)
        logger.info(f"Validated {fpath}")
