from typing import List, Iterable, Tuple, Generator
import itertools
from os import path

from context import Context, LabelNode, Label, QA, decorate, contextify, jsonify, contextify_rd, yellow, red, blue
from data import DATA_PATH, json_loader, count_contexts, JsonLoader, json_dumper
from list_utils import find_subseq, find_all, find_subseq_spaced


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


#---- QA pairs gen rules ----


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
            ners = find_all(context.ner, lambda entity: entity in np_subj)
            if len(ners) == 1:
                ner_label = context.ner[ners[0]].copy(color="green")
                children = [node.copy_no_child(color="magenta") for node in sent_const.children[idx: idx+3]]
                cloze = LabelNode(start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red")
                ret.append((cloze, ner_label))
    return ret

def rule1_ext(context: Context) -> Rule1_RT:
    """Extract 'NP-SUJ', 'VN', 'NP-ATS' constituents sub-sequence in sentence if NP-SUJ contains a single named entity.
    
    Returns
    -------
        List[Tuple[LabelNode, Label]]
        A list of pairs; The pair f
    """
    ret = list()
    for sent_const in context.constituents:
        indices = find_subseq_spaced([c.label for c in sent_const.children], ['NP-SUJ', 'VN', 'NP-ATS'])
        if indices:
            np_subj = sent_const.children[indices[0]]
            ners = find_all(context.ner, lambda entity: entity in np_subj)
            if len(ners) == 1:
                ner_label = context.ner[ners[0]].copy(color="green")
                children = [sent_const.children[i].copy_no_child(color="magenta") for i in indices]
                
                # extends NP-ATS if followed by PP-MOD (and PP-MOD starts with 'à' ???)
                next_label_idx = indices[-1] + 1
                next_label = sent_const.children[next_label_idx] if len(sent_const.children) > next_label_idx else None
                if next_label and next_label.label.startswith("PP"):  # and next_label.extract(context.text)[0] == 'à':
                    children[-1].end = next_label.end
                    children[-1].label = f"NP-ATS + {next_label.label}"
                    children[-1].extras["color"] = 'red'
                
                cloze = LabelNode(start=children[0].start, end=children[-1].end, label="CQ", children=children, color="red")
                ret.append((cloze, ner_label))
    return ret


def rule1_to_qa(context: Context, filter1_ret: Rule1_RT) -> None:
    """Convert rule1 result to a question answer pair."""
    for el in filter1_ret:
        suj, vn, ats  = el[0].children
        vn_txt, ats_txt = vn.extract(context.text), ats.extract(context.text)
        ner_label = el[1].label

        # qword = "Quel" if use_qword_quel(ats_txt) else NER_TO_QWORD[ner_label]
        qword = "Quel"
        question = " ".join([qword, vn_txt, ats_txt, "?"])
        answer = suj 
        # if qword == "Quel":
        context.qas.append(QA(question, answer))


def show_rule1(context_it: Iterable[Context], show_other_d1_const=True):
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
            print(decorate(context.text, labels))
            for question, answer in context.qas:
                print(blue(question))
                print(answer.extract(context.text))
            input()

#---- QA pairs generation functions ----


def generate_qas(context_it: Iterable[Context], filter_no_qa:bool = True) -> Generator[Context, None, None]:
    for context in context_it:
        ret = rule1_ext(context)
        rule1_to_qa(context, ret)
        if context.qas or not filter_no_qa:
            yield context


def to_squad_format(fpath:str, context_it: Iterable[Context], version:str='0.1'):
    qid_cnt = 0
    fcontent = dict(version=version)
    data_dict = dict()
    qa_count = 0
    for context in context_it:
        if context.doc_id not in data_dict:
            data_dict[context.doc_id] = dict(
                title=context.doc_title,
                paragraphs=list()
            )
        
        context_dict = dict(qas=list(), context=context.text)
        for qa in context.qas:
            qa_dict = dict(
                question=qa.question,
                id=qid_cnt,
                answers=[{"text": qa.answer.extract(context.text), "answer_start": qa.answer.start}]
            )
            qid_cnt += 1
            context_dict["qas"].append(qa_dict)
        data_dict[context.doc_id]["paragraphs"].append(context_dict)
        qa_count += len(context.qas)
    data = [v for k, v in sorted(data_dict.items())]
    fcontent["data"] = data
    print("Total number of generated question/answer pairs:", qa_count)
    yield fpath, fcontent


#---- Others ----

def show_const_ner_depth(context_it, depth=1):
    for context in context_it:
        print(yellow(context.fpath))
        print(yellow(f'{context.doc_id} - {context.doc_title} [{context.context_id}]'))
        context.set_color_hier("constituents", colors=["white", "red"])
        labels = [el.copy(depth=depth) for el in context.constituents]
        context.set_color_all("ner", 'green')
        labels.extend(context.ner)
        print(decorate(context.text, labels))
        input("\n")

if __name__ == "__main__":
    from data import json_opener, add_suffix
    # fpath = path.join(DATA_PATH, "fsquad_ner_const.json")
    # outfpath = path.join(DATA_PATH, "fsquad_gen.json")
    # for fpath in json_dumper(to_squad_format(outfpath ,generate_qas(contextify(json_opener((fpath,)))), '0.3')):
    #     print(f"Saved {fpath}")
    # exit()

    data_dir = path.join(DATA_PATH, "ga_const")

    # show_rule1(contextify_rd(JsonLoader(data_dir, sort_by_filename=False)))
    
    outpath = path.join(DATA_PATH, "train_0-3.json")
    for fpath in json_dumper(to_squad_format(outpath, generate_qas(contextify(JsonLoader(data_dir))), '0.3')):
        print(f"Saved {fpath}")
   