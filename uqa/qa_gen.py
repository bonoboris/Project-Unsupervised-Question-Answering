"""Natural question / answer pairs generation.

Using output of `ner` and `constituency parsing` steps this module implements
rules to generate question / answer pairs in natural looking French.

Notes
-----
For visualization purposes question / anwser pairs generation process is split in 2 steps:

    1. Matching interesting patterns in sentences constituency and / or named entity
       (see :func:`rule1` and :func:`rule1_ext`)
    2. Generating the questions answers from those patterns (see :func:`rule1_to_qa`)
"""

from typing import List, Iterable, Tuple
import itertools

from uqa import context_utils, list_utils, dataset


# ---- Question making helpers ----


NER_TO_QWORD = {"PER": "Qui", "LOC": "Où", "MISC": "Qu'est-ce que", "ORG": "Qu'est-ce que"}

SUPERLATIVES = [
    "le plus",
    "la plus",
    "les plus",
    "le moins",
    "la moins",
    "les moins",
    "le principal",
    "la principale",
    "les principales",
]

_ORDINALS = [
    "premier",
    "première",
    "second",
    "deuxième",
    "troisième",
    "quatrième",
    "cinquième",
    "sixième",
    "septième",
    "huitième",
    "neuvième",
    "dixième",
    "onzième",
    "douzième",
]

ORDINALS = list()
for _ord in _ORDINALS:
    ORDINALS.append(f"le {_ord}")
    ORDINALS.append(f"les {_ord}s")
    if _ord.endswith("e"):
        ORDINALS.append(f"la {_ord}")
    else:
        ORDINALS.append(f"la {_ord}e")
        ORDINALS.append(f"les {_ord}es")


def use_qword_quel(txt: str) -> bool:
    """(unused). Analyse `txt` and returns ``True`` if the generated question should use 'quel' question word.

    Returns ``True`` if a superlative or an ordinal is found in the sentence.
    """
    for word in itertools.chain(SUPERLATIVES, ORDINALS):
        if word in txt:
            return True
    return False


# ---- QA pairs gen rules ----

#: Return type of :func:`rule1` and :func:`rule1_ext`
Rule1_RT = List[Tuple[context_utils.LabelNode, context_utils.Label]]  # pylint: disable=invalid-name


def rule1(context: context_utils.Context) -> Rule1_RT:
    """Extract `'NP-SUJ'`, `'VN'`, `'NP-ATS'` constituents continuous subsequence in sentence
    if `'NP-SUJ'` contains a single named entity.

    Returns
    -------
    :obj:`Rule1_RT`
        A list of pairs. Each pair first element is a :class:`LabelNode`
        with `'NP-SUJ'`, `'VN'`, `'NP-ATS'` labels as children and second element is the `NER` label.
    """
    ret = list()
    for sent_const in context.constituents:
        idx = list_utils.find_subseq([c.label for c in sent_const.children], ["NP-SUJ", "VN", "NP-ATS"])
        if idx > -1:
            np_subj = sent_const.children[idx]
            ners = list_utils.find_all(context.ner, lambda entity, label=np_subj: entity in label)
            if len(ners) == 1:
                ner_label = context.ner[ners[0]].copy(color="green")
                children = [node.copy_no_child(color="magenta") for node in sent_const.children[idx : idx + 3]]
                cloze = context_utils.LabelNode(
                    start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red"
                )
                ret.append((cloze, ner_label))
    return ret


def rule1_ext(context: context_utils.Context) -> Rule1_RT:
    """Extract `'NP-SUJ'` -- `'VN'` -- `'NP-ATS'` or `'NP-SUJ'` -- `'VN'` -- `'NP-ATS + PP'`
    constituents subsequence in sentence if `'NP-SUJ'` contains a single named entity.

    | Here subsequence corresponds with the mathematical definition (see :func:`.find_subseq_spaced`)
    | If `'NP-ATS'` is followed by a `'PP'` constituent `'NP-ATS'` and `'PP'` are merged into a single constituent.

    Returns
    -------
    :obj:`Rule1_RT`
        A list of pairs. Each pair first element is a :class:`LabelNode`
        with constituents labels as children and second element is the `NER` label.
    """
    ret = list()
    for sent_const in context.constituents:
        indices = list_utils.find_subseq_spaced([c.label for c in sent_const.children], ["NP-SUJ", "VN", "NP-ATS"])
        if indices:
            np_subj = sent_const.children[indices[0]]
            ners = list_utils.find_all(context.ner, lambda entity, label=np_subj: entity in label)
            if len(ners) == 1:
                ner_label = context.ner[ners[0]].copy(color="green")
                children = [sent_const.children[i].copy_no_child(color="magenta") for i in indices]

                # extends NP-ATS if followed by PP-MOD (and PP-MOD starts with 'à' ???)
                next_label_idx = indices[-1] + 1
                next_label = sent_const.children[next_label_idx] if len(sent_const.children) > next_label_idx else None
                if next_label and next_label.label.startswith("PP"):  # and next_label.extract(context.text)[0] == 'à':
                    children[-1].end = next_label.end
                    children[-1].label = f"NP-ATS + {next_label.label}"
                    children[-1].extras["color"] = "red"

                cloze = context_utils.LabelNode(
                    start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red"
                )
                ret.append((cloze, ner_label))
    return ret


def rule1_to_qa(context: context_utils.Context, rule1_ret: Rule1_RT) -> None:
    """Generate question / answer pairs from either :func:`rule1` or :func:`rule1_ext` outputs.

    Parameters
    ---------
    context: :class:`.Context`
        Context instance for which `rule1_ret` is derived, generated :class:`.QA` are stored in
        `context` :attr:`~.Context.qas` attribute.
    rule1_ret: :obj:`Rule1_RT`
        The ouptut of either :func:`rule1` or :func:`rule1_ext` applied to `context`
    """
    for match in rule1_ret:
        suj, vn, ats = match[0].children  # pylint: disable=invalid-name
        vn_txt, ats_txt = vn.extract(context.text), ats.extract(context.text)
        # ner_label = el[1].label

        # qword = "Quel" if use_qword_quel(ats_txt) else NER_TO_QWORD[ner_label]
        qword = "Quel"
        question = " ".join([qword, vn_txt, ats_txt, "?"])
        answer = suj
        # if qword == "Quel":
        context.qas.append(context_utils.QA(question, answer.to_label()))


# def rule2(context: context_utils.Context) -> Rule1_RT:
#     """Extract 'NP-SUJ', 'VN', 'AP-ATS' constituents sub-sequence in sentence if NP-SUJ contains a single named entity.

#     Returns
#     -------
#         List[Tuple[LabelNode, Label]]
#         A list of pairs; The pair f
#     """
#     ret = list()
#     for sent_const in context.constituents:
#         idx = list_utils.find_subseq([c.label for c in sent_const.children], ["NP-SUJ", "VN", "AP-ATS"])
#         if idx > -1:
#             np_subj = sent_const.children[idx]
#             ners = list_utils.find_all(context.ner, lambda entity, label=np_subj: entity in label)
#             if len(ners) == 1:
#                 ner_label = context.ner[ners[0]].copy(color="green")
#                 children = [node.copy_no_child(color="magenta") for node in sent_const.children[idx : idx + 3]]
#                 cloze = context_utils.LabelNode(
#                     start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red"
#                 )
#                 ret.append((cloze, ner_label))
#     return ret

# ---- QA pairs generation functions ----


def generate_qas_context_it(
    context_it: Iterable[context_utils.Context], filter_no_qa: bool = True
) -> Iterable[context_utils.Context]:
    """Generate question / answers pairs on a context iterable.

    NER and constituency parsing steps must have been realized prior to q/a generation.

    Parameters
    ----------
    data_it: Iterable[:class:`.Context`]
        A context iterable.

    Returns
    -------
    :obj:`.DataIterble`
        The processed dateset iterable.
    """
    for context in context_it:
        ret = rule1_ext(context)
        rule1_to_qa(context, ret)
        if context.qas or not filter_no_qa:
            yield context


def generate_qas_dl(data_it: dataset.DataIterable) -> dataset.DataIterable:
    """Generate question / answers pairs on a dataset.

    NER and constituency parsing steps must have been realized prior to q/a generation.

    Parameters
    ----------
    data_it: :obj:`.DataIterble`
        A dateset iterable in `default` format.

    Returns
    -------
    :obj:`.DataIterble`
        The processed dateset iterable.
    """
    yield from context_utils.jsonify(generate_qas_context_it(context_utils.contextify(data_it)))
