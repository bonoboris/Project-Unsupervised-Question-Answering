import bz2
import re
import json
import os


def wiki_extractor_parser():
    rootdir = 'wiki_dumps_fr'
    for subdir, dirs, files in os.walk(rootdir):
        print(subdir, dirs)
        for file in files:
            path_file = os.path.join(subdir, file)
            with bz2.open(path_file, 'rb') as file:
                list_json_documents = file.readlines()
                file_json = list()
                for string_json in list_json_documents:
                    # Getting the Parsed Json
                    string_json_decoded = string_json.decode(encoding='utf-8')
                    wiki_article_json = json.loads(string_json_decoded)
                    # Parsing the Text into Paragraph
                    article_text = wiki_article_json['text']
                    article_text = re.sub(r'\n+', '\n', article_text)
                    contexts = article_text.split('\n')
                    # Remove last and first element
                    contexts.pop(0)
                    contexts.pop(-1)
                    final_contexts = []
                    for id_context, context in enumerate(contexts):
                        final_contexts.append({
                            "id_context": id_context,
                            "text": context
                        })
                    final_json_list = {
                        'id_doc': int(wiki_article_json['id']),
                        'title': wiki_article_json['title'],
                        'contexts': final_contexts
                    }
                    file_json.append(final_json_list)
                yield path_file, file_json


if __name__ == '__main__':
    print("check")