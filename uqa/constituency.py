from collections import Counter
from typing import Iterable, Tuple, Any

from benepar.spacy_plugin import BeneparComponent
import spacy
from tqdm import tqdm
from tabulate import tabulate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)
    pass

from context import Context, LabelNode
from spacywrapper import SpacyFrenchModelWrapper as Model

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


def constituency_fcontent(fcontent):
    """Must add BeneparComponent to the pipe prior to the function call"""
    for article in fcontent:
        for context in tqdm(article["contexts"], unit="context"):
            spacy_doc = Model.model(context["text"], disable=["ner", "tagger"])
            consts_json = [span_to_node(sent).to_json() for sent in spacy_doc.sents]
            context["constituency"] = consts_json
    return fcontent

def constituency_json(json_file_it: Iterable[Tuple[str, JsonT]]):
    Model.model.add_pipe(BeneparComponent("benepar_fr"))
    for fpath, fcontent in tqdm(json_file_it, unit="file"):
        print(f"Processing {fpath}")
        yield fpath, constituency_fcontent(fcontent)


def mp_consituency(dirpath, new_dir, max_workers=8):
    dirs = dir_path.split("/")
    new_dir = path.join(*dirs[:-1], new_dir)
    
    def init(lock: Lock):
        return
        print("--- INIT ---")
        with lock:
            print("--- LOCKED ---")
            Model.model.add_pipe(BeneparComponent("benepar_fr"))
            print("--- UNLOCK ---")
        print("--- INIT FINISHED ---")

    def work(fpath):
        print(f"Processing {fpath}")
        return
        fcontent = None
        with open(fpath, 'r', encoding='utf8') as file:
            fcontent = json.load(file.read())
        new_fcontent = constituency_fcontent(fcontent)
        sub_path = path.relpath(fpath, start=dir_path)
        new_path = path.join(new_dir, sub_path)
        with open(new_path, 'w', encoding='utf8') as file:
            json.dump(new_fcontent, file)
        print(f"Saved {new_path}")
        return new_path
    
    def work_ce(fpath):
        print("in work ce")
        try:
            work(fpath)
        except Exception as e:
            print(f"While processing {fpath}:\n{e}")
            pass

    all_path_it = json_discover(dirpath)
    lock = Lock()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init, initargs=(lock,)) as executor:
        print('----- WORKING -----')
        res = list(executor.map(work, all_path_it))
    return res


def len_context_test(json_file_it):
    context_len_count = Counter()
    for fpath, json_content in tqdm(json_file_it, unit="file"):
        print(f"Processing file {fpath}")
        for article in tqdm(json_content, unit='articles'):
            for context in article["contexts"]:
                context_len_count[len(context["text"])//10] += 1
                if len(context["text"]) < 10:
                    print(context)
                    print()
    return context_len_count


def len_sent_test(json_file_it):
    sent_len_count = Counter()
    for fpath, json_content in tqdm(json_file_it, unit="file"):
        print(f"Processing file {fpath}")
        for article in tqdm(json_content, unit='articles'):
            for context in article["contexts"]:
                spacy_doc = Model.model(context["text"], disable=["ner", "tagger"])
                for sent in spacy_doc.sents:
                    sent_len_count[len(sent)] += 1
    return sent_len_count


from argparse import ArgumentParser
job_array_parser = ArgumentParser(
    description="""Launch named constituency parssing as a job array, each job will handle a part of the work.
                This only work for multifile database.""")
job_array_parser.add_argument("job_index", type=int, help="The job index")
job_array_parser.add_argument("num_jobs", type=int, help="The total number of jobs")
job_array_parser.add_argument("dirpath", type=str, help="The directory containing the dataset")
job_array_parser.add_argument("output_dirname", type=str, nargs="?", help="(Optional) The output directory name, if left empty the output dir name will be the input directoty name followed by '_const'.")
job_array_parser.add_argument("-O","--override", type=bool, default=False, help="[default: False] Override existing output file. If false, raise an error if trying to override existing file.")


if __name__ == "__main__":
    from data import DATA_PATH, json_loader, change_dir, json_dumper, pickle_loader, json_discover, json_opener
    from os import path
    from context import contextify, jsonify
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Lock
    import json

    from data import json_discover, json_opener, json_dumper, add_suffix, change_dir, change_last_dir
    from list_utils import split_chunks

    args = job_array_parser.parse_args()
    files = list(sorted(json_discover(args.dirpath)))
    files_chunk = list(split_chunks(files, args.num_jobs))[args.job_index]
    dirname = args.dirpath.rstrip("/").split("/")[-1]
    if args.output_dirname:
        out_dirname = args.output_dirname
    else:
        out_dirname = dirname + '_const'
    out_dirpath = change_last_dir(args.dirpath, out_dirname)

    for fpath in json_dumper(change_dir(constituency_json(json_opener(files_chunk)), args.dirpath, out_dirpath), override=args.override):
        print(f"Saved {fpath}")
    exit()

    
    # dir_path = path.join(DATA_PATH, "ga_json_small")
    # res = mp_consituency(dir_path, "ga_const_json_small", 2)

    # print(len(res), "file saved")
    # fpath = path.join(DATA_PATH, "good_articles.pickle")
    # len_context_test(pickle_loader(fpath))

    # dir_path = path.join(DATA_PATH, "ner_json")
    # for saved_path in json_dumper(change_dir(constituency_json(json_loader(dir_path)), "ner_json", "test_const"), override=True):
    #     print(saved_path)
    
