from typing import Iterable, Tuple, Any

from benepar.spacy_plugin import BeneparComponent
import spacy

from context import Context, LabelNode
from spacywrapper import SpacyFrenchModelWrapper as Model
from tqdm import tqdm

JsonT = Any


def span_to_node(span: spacy.tokens.Span) -> LabelNode:
    node = LabelNode(span.start_char, span.end_char, "|".join(span._.labels))
    node.children = [span_to_node(child) for child in span._.children]
    return node


def constituency_context(context_it: Iterable[Context]):
    Model.model.add_pipe(BeneparComponent("benepar_fr"))
    for context in context_it:
        spacy_doc = Model.model(context.text, disable=["ner", "tagger"])
        consts = list()
        for sent in spacy_doc.sents:
            consts.append(span_to_node(sent))
        context.constituents = consts
        yield context


def constituency_json(json_file_it: Iterable[Tuple[str, JsonT]]):
    Model.model.add_pipe(BeneparComponent("benepar_fr"))
    for fpath, json_content in tqdm(json_file_it, unit="file"):
        for article in json_content:
            for context in tqdm(article["contexts"], unit="context"):
                spacy_doc = Model.model(context["text"], disable=["ner", "tagger"])
                consts_json = [span_to_node(sent).to_json() for sent in spacy_doc.sents]
                context["constituency"] = consts_json
        yield fpath, json_content 


if __name__ == "__main__":
    from data import DATA_PATH, json_loader, change_dir, json_dumper
    from os import path
    from context import contextify, jsonify

    spacy.prefer_gpu()
    #spacy.require_gpu()

    dir_path = path.join(DATA_PATH, "ner_json")
    for saved_path in json_dumper(change_dir(constituency_json(json_loader(dir_path)), "ner_json", "test_const"), override=True):
        print(saved_path)
    
