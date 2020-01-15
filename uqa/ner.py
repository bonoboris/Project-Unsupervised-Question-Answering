import itertools

from spacywrapper import SpacyFrenchModelWrapper


test_data = [
    [
        {
            "id_doc": 0,
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
    context_it = (context["text"] for json_file in json_file_it_copy for article in json_file for context in article["contexts"])
    docs_it = SpacyFrenchModelWrapper.model.pipe(context_it, disable=["parser", "tagger"])
    for num_file, json_file in enumerate(json_file_it):
        print(f"File {num_file}")
        for num_article, json_article in enumerate(json_file):
            print(f"Article {num_article} with id {json_article['id_doc']}")
            for num_context, json_context in enumerate(json_article["contexts"]):
                print(f"Context {num_context} with id {json_context['id_context']}")
                doc = next(docs_it)
                print(doc.text)
                json_context["ents"] = doc.to_json()["ents"]
        yield json_file


def main():
    for json_like in ner_gen(test_data):
        print(json_like)


if __name__ == '__main__':
    main()
