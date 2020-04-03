from typing import List, Iterable, Tuple, Generator
import itertools

from context import Context, LabelNode, Label, QA, decorate, yellow
from list_utils import find_subseq, find_all


#---- Question making helpers ----

NER_TO_QWORD = {
    "PER": "Qui",
    "LOC": "Où",
    "MISC": "Qu'est-ce que",
    "ORG": "Qu'est-ce que"
}

superlatives = [
    "le plus", "la plus", "les plus", "le moins", "la moins", "les moins", "le principal", "la principale", "les principales"
]

_ordinals = [
    "premier", "première", "second", "deuxième", "troisième", "quatrième", "cinquième", "sixième",
    "septième", "huitième", "neuvième", "dixième", "onzième", "douzième"
]

ordinals = list()
for _ord in _ordinals:
    ordinals.append(f"le {_ord}")
    ordinals.append(f"les {_ord}s")
    if _ord.endswith("e"):
        ordinals.append(f"la {_ord}")
    else:
        ordinals.append(f"la {_ord}e")
        ordinals.append(f"les {_ord}es")

others_quel = [

]

def use_qword_quel(txt: str) -> bool:
    for el in itertools.chain(superlatives, ordinals):
        if el in txt:
            return True
    return False 

#---- QA pairs generation ----

Rule1_RT = List[Tuple[LabelNode, Label]]

def rule1(context: Context) -> Rule1_RT:
    """Extract 'NP-SUJ', 'VN', 'NP-ATS' constituents sub-sequence in sentence if NP-SUJ contains a single named entity.
    
    Returns
    -------
        List[Tuple[LabelNode, Label]]
        A list of pairs; The pair f
    """
    ret = list()
    for sent_const in context.constituents:
        idx = find_subseq([c.label for c in sent_const.children], ['NP-SUJ', 'VN', 'NP-ATS'])
        if idx > -1:
            np_subj = sent_const.children[idx]
            ners = find_all(context.ner, lambda el: el in np_subj)
            if len(ners) == 1:
                ner_label = context.ner[ners[0]].copy(color="green")
                children = [node.copy_no_child(color="magenta") for node in sent_const.children[idx: idx+3]]
                cloze = LabelNode(start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red")
                ret.append((cloze, ner_label))
    return ret

def rule1_to_qa(context: Context, filter1_ret: Rule1_RT) -> None:
    """Convert rule1 result to a question answer pair."""
    for el in filter1_ret:
        suj, vn, ats  = el[0].children
        vn_txt, ats_txt = vn.extract(context.text), ats.extract(context.text)
        ner_label = el[1].label

        qword = "Quel" if use_qword_quel(ats_txt) else NER_TO_QWORD[ner_label]
        question = " ".join([qword, vn_txt, ats_txt, "?"])
        answer = suj 
        # if qword == "Quel":
        context.qas.append(QA(question, answer))


def generate_qas(context_it: Iterable[Context], filter_no_qa:bool = True) -> Generator[Context, None, None]:
    for context in context_it:
        ret = rule1(context)
        rule1_to_qa(context, ret)
        if context.qas or not filter_no_qa:
            print(decorate(context.text, ret))
            for question, answer_label in context.qas:
                print(yellow(question))
                print(answer_label.extract(context.text))
            input()
            yield context


if __name__ == "__main__":
    from os import path
    from data import DATA_PATH, json_loader
    from context import contextify

    data_dir = path.join(DATA_PATH, "const_json")
    cl = contextify(json_loader(data_dir))
    cnt = 0
    for context in generate_qas(cl):
        cnt += len(context.qas)
    print("Generated questions:", cnt)