from utils import json_loader, DATA_PATH, JsonLoader, json_opener, json_dumper
from os import path

def validate(json_file_it):
    for fpath, fcontent in json_file_it:
        data = fcontent["data"]
        for article in data:
            for para in article["paragraphs"]:
                context = para["context"]
                qas = para["qas"]
                for qa in qas:
                    ans = qa["answers"][0]
                    ans_start = ans["answer_start"]
                    ans_text = ans["text"]
                    ans_len = len(ans_text)
                    assert len(context) >= ans_start + ans_len
                    assert ans["text"] == context[ans_start: ans_start + ans_len]
        print(f"Validated {fpath}")


# def show(json_file_it, show_context=False):
#     for fpath, fcontent in json_file_it:
#         data = fcontent["data"]
#         for article in data:
#             for para in article["paragraphs"]:
#                 context = para["context"]
#                 qas = para["qas"]
#                 for qa in qas:
#                     ans = qa["answers"][0]
#                     ans_start = ans["answer_start"]
#                     ans_text = ans["text"]
#                     ans_len = len(ans_text)
#                     assert len(context) >= ans_start + ans_len
#                     assert ans["text"] == context[ans_start: ans_start + ans_len]


def convert_fquad_into_good_format(json_file_it):
    for fpath, fcontent in json_file_it:
        articles = fcontent["data"]
        output = []
        for index_article, article in enumerate(articles):
            context_id = 0
            paragraphs = article['paragraphs']
            title = article['title']
            contexts = []
            for paragraph in paragraphs:
                context = paragraph['context']
                contexts.append({"id_context": context_id, "text": context})
                context_id += 1
            doc_info = {
                "id_article": index_article,
                "title": title,
                "contexts": contexts
            }
            output.append(doc_info)
        yield fpath, output

def count_qa()

if __name__ == "__main__":
    data_file = path.join(DATA_PATH, "fsquad_train.json")
    fpath, fcontent = next(convert_fquad_into_good_format(json_opener((data_file,))))
    base, filename = path.split(fpath)
    new_fpath = path.join(base, "fsquad.json")
    for fpath in json_dumper(((new_fpath, fcontent),)):
        print(f"Saved {fpath}")