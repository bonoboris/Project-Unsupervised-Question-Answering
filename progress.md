Table
======

Json format
-----------
- One file per `wiki dump` file
```
[
    {
        "doc_id": 0  # unique id in the corpus,
        "article_title": "Jacques Chirac",
        "context":[
            {
                "context_id": 0  # unique in the article
                "text" : "bla bla bla..."
            },
            {
                ...
            },
            ...
        ]
    },
    {
        ...
    },
    ...
]
```

Named entity recognition
------------------------
### Key points

- Reading `wiki dumps` file with function in [reading_wiki_dumps.py] 
- Using [spacy](https://spacy.io/models/fr#fr_core_news_md) `fr_core_news_md` model
    French multi-task CNN trained on the French Sequoia (Universal Dependencies) and WikiNER corpus. 
- Only 4 labels: `PERS`, `LOC`, `ORG`, `MISC`
- Saving in `json` format: Adding to each entry in each context:
```
"entities": [
    {
        "start": 12  # character index start (excluded),
        "end": 24 # character index end (excluded),
        "label": "PERS,
    },
    ...
]
```

### Problem

- Not state of the art model ? (fine-tuning Camembert should yield better perf)
- Too few NER labels, missing especially Date / Temporal labelling

Constituency Parsing
--------------------

### Key points

- Reading Json formated dumps
- Using [benepar](https://pypi.org/project/benepar/) french constituency parser
    Self-attentive encoder-decoder architecture, trained on French Treebank
- Integrate in `scacy` pipeline.
- Saving in `json` format in tree-like structre:
```
"constituency": [  # One root node per sentence in the context with label `SENT`
    { # firt level children -> sentences
        "start": 0,
        "end": 243,
        "label": "SENT",
        "children": [  # second level children -> depth-1 constituents
            {
                "start": 0,
                "end": 40,
                "label": "NP-SUJ"
                "children": [
                    ...
                ],
                ...
            }
        ]
    },
    ...
]
```

### Visualization

Feature developped to decorate the context `text` with `entities` and `constituency` json-like data
- Use [colorama](https://pypi.org/project/colorama/) to colorize the standard output for visibility.

### Analysis, Cloze question extraction & Natural question generation

Rule based approach
Found patterns:
- NP-SUJ / VN / NP-ATS: NP-SUJ contains a single NER -> cloze question: the 3 constituents -> the answer the named-entity.

#### Glossary

##### Consistuent annotations

Name | Signification
-----|--------------
NP | "syntagme nominal": groupe nominal
VN | "noyau verbal": groupe verbale principale

##### Functional annotations

Name | Signification
-----|--------------
*-SUJ | sujet
*-ATS | attribut du sujet

### Problems

- For now the script using `benepar` constituency parser runs on the CPU and is quite slow: ETA 10 hours to parse 200 wiki dump files (13MB)
- The current wiki dumps contains a lot of incomplete sentences which may hinder the constituency parser performances.