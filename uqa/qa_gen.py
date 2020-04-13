"""Natural question / answer pairs gneration"""

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
    """Wether to use 'quel' question word."""
    for word in itertools.chain(SUPERLATIVES, ORDINALS):
        if word in txt:
            return True
    return False


# ---- QA pairs gen rules ----


Rule1_RT = List[Tuple[context_utils.LabelNode, context_utils.Label]]  # pylint: disable=invalid-name


def rule1(context: context_utils.Context) -> Rule1_RT:
    """Extract 'NP-SUJ', 'VN', 'NP-ATS' constituents sub-sequence in sentence if NP-SUJ contains a single named entity.

    Returns
    -------
        List[Tuple[LabelNode, Label]]
        A list of pairs; The pair f
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
    """Extract 'NP-SUJ', 'VN', 'NP-ATS' constituents sub-sequence in sentence if NP-SUJ contains a single named entity.

    Returns
    -------
        List[Tuple[LabelNode, Label]]
        A list of pairs; The pair f
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


def rule1_to_qa(context: context_utils.Context, filter1_ret: Rule1_RT) -> None:
    """Convert rule1 result to a question answer pair."""
    for match in filter1_ret:
        suj, vn, ats = match[0].children  # pylint: disable=invalid-name
        vn_txt, ats_txt = vn.extract(context.text), ats.extract(context.text)
        # ner_label = el[1].label

        # qword = "Quel" if use_qword_quel(ats_txt) else NER_TO_QWORD[ner_label]
        qword = "Quel"
        question = " ".join([qword, vn_txt, ats_txt, "?"])
        answer = suj
        # if qword == "Quel":
        context.qas.append(context_utils.QA(question, answer.to_label()))


def show_rule1(context_it: Iterable[context_utils.Context], show_other_d1_const=True):
    """Show rule1 matchs."""
    for context in context_it:
        ret = rule1_ext(context)
        rule1_to_qa(context, ret)
        if context.qas:
            print(context.header(color="yellow"))
            labels = list()
            for cloze, ner_label in ret:
                labels.extend(cloze.children)
                labels.append(ner_label)
            if show_other_d1_const:
                for cloze, _ in ret:
                    sent_label = [l for l in context.constituents if cloze in l][0]
                    labels.append(sent_label.copy_no_child(color="white"))
                    labels.extend((l.copy_no_child(color="cyan") for l in sent_label.children if l not in labels))
            print(context.decorate(context.text, labels))
            for question, answer in context.qas:
                print(context.blue(question))
                print(answer.extract(context.text))
            input()


# ---- QA pairs generation functions ----


def generate_qas_context_it(
    context_it: Iterable[context_utils.Context], filter_no_qa: bool = True
) -> Iterable[context_utils.Context]:
    """From a context_utils.Context iterable, generate natural question / answer pairs."""
    for context in context_it:
        ret = rule1_ext(context)
        rule1_to_qa(context, ret)
        if context.qas or not filter_no_qa:
            yield context


def generate_qas_dl(data_it: dataset.DataIterable) -> dataset.DataIterable:
    """For a default structure dataset, generate natural question answer pairs."""
    yield from context_utils.jsonify(generate_qas_context_it(context_utils.contextify(data_it)))


# ---- Others ----


def show_const_ner_depth(context_it, depth=1):
    """Display colorized named entities and constituents at `depth` depth and less."""
    for context in context_it:
        print(context_utils.yellow(context.fpath))
        print(context_utils.yellow(f"{context.doc_id} - {context.doc_title} [{context.context_id}]"))
        context.set_color_hier("constituents", colors=["white", "red"])
        labels = [el.copy(depth=depth) for el in context.constituents]
        context.set_color_all("ner", "green")
        labels.extend(context.ner)
        print(context_utils.decorate(context.text, labels))
        input("\n")
