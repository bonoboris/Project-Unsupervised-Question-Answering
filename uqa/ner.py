import itertools
import json
from os import path, makedirs
from encodings.utf_8 import decode, encode

from spacywrapper import SpacyFrenchModelWrapper
from tqdm import tqdm

from data import DATA_PATH, count_contexts, pickle_loader, pickle_dumper, add_suffix
from reading_wiki_dumps import wiki_extractor_parser

test_data = [
    [
        {
            "id_doc": 0,
            "title": "FOOOOO",
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 0."
                },
                {
                    "id_context": 1,
                    "text": "Je suis à Paris avec mon ami Jacques."
                },
            ]
        },
        {
            "id_doc": 1,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 1."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 2."
                },
            ]
        }
    ],
    [
        {
            "id_doc": 2,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 3."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 4."
                },
            ]
        },
        {
            "id_doc": 3,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 5."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 6."
                },
            ]
        }
    ],
    [
        {
            "id_doc": 3,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 7."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 8."
                },
            ]
        },
        {
            "id_doc": 4,
            "contexts": [
                {
                    "id_context": 0,
                    "text": "Je suis le context 9."
                },
                {
                    "id_context": 1,
                    "text": "Je suis le context 10."
                },
            ]
        }
    ],
]


def ner_gen(json_file_it):
    json_file_it, json_file_it_copy = itertools.tee(json_file_it)
    context_it = (context["text"] for _, json_file in json_file_it_copy for article in json_file for context in article["contexts"])
    docs_it = SpacyFrenchModelWrapper.model.pipe(context_it, disable=["parser", "tagger"])
    for file_path, json_file in json_file_it:
        print(f"Processing file {file_path}")
        tbar = tqdm(unit="contexts", total=count_contexts(json_file))
        for json_article in json_file:
            for json_context in json_article["contexts"]:
                doc = next(docs_it)
                json_context["entities"] = doc.to_json()["ents"]
                tbar.update()
        yield file_path, json_file


NER_DATA_FOLDER = "ner_json"


def generate_ner_json():
    prefix = path.join(DATA_PATH, "ner_json")
    for filepath, ner_json in ner_gen(wiki_extractor_parser()):
        print(f"{filepath} done...", end=" ")
        subdirs, filename = path.split(filepath)
        filename = ".".join(filename.split(".")[:-1]) + ".json"
        dstpath = path.join(prefix, subdirs, filename)
        if not path.exists(path.dirname(dstpath)):
            makedirs(path.dirname(dstpath))
        with open(dstpath, "w") as file:
            json.dump(ner_json, file, indent=None)
            print("saved !")


def ner_pickled(filepath):
    for fpath, fcontent in ner_gen(pickle_loader(filepath)):
        dirpath, filename = path.split(fpath)
        name, ext = path.splitext(filename)
        new_fpath = path.join(dirpath, f"{name}_ner{ext}")
        yield new_fpath, fcontent


from argparse import ArgumentParser
job_array_parser = ArgumentParser(
    description="""Launch named entity recognition as a job array, each job will handle a part of the work.
                This only work for multifile database.""")
job_array_parser.add_argument("job_index", type=int, help="The job index")
job_array_parser.add_argument("num_jobs", type=int, help="The total number of jobs")
job_array_parser.add_argument("dirpath", type=str, help="The directory containing the dataset")
job_array_parser.add_argument("output_dirname", type=str, nargs="?", help="(Optional) The output directory name, if left empty the output dir name will be the input directoty name followed by '_ner'.")
job_array_parser.add_argument("-O","--override", type=bool, default=False, help="[default: False] Override existing output file. If false, raise an error if tryign to override existing file.")


if __name__ == '__main__':
    from data import json_discover, json_opener, json_dumper, add_suffix, change_dir, change_last_dir
    from list_utils import split_chunks

    args = job_array_parser.parse_args()
    files = list(sorted(json_discover(args.dirpath)))
    files_chunk = list(split_chunks(files, args.num_jobs))[args.job_index]
    dirname = args.dirpath.rstrip("/").split("/")[-1]
    if args.output_dirname:
        out_dirname = args.output_dirname
    else:
        out_dirname = dirname + '_ner'
    out_dirpath = change_last_dir(args.dirpath, out_dirname)

    for fpath in json_dumper(change_dir(ner_gen(json_opener(files_chunk)), args.dirpath, out_dirpath), override=args.override):
        print(f"Saved {fpath}")
    exit()


    filepath = path.join(DATA_PATH, "good_articles_small.pickle")
    for fpath in pickle_dumper(add_suffix(ner_gen(pickle_loader(filepath)), "_ner")):
        print(f"Saved {fpath}")
