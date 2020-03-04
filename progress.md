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
- Quoting charcher often labeled as ORG
- inline formula text (ex: 'formula_5') in mathematical articles often mislabeled

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

### Analysis, Cloze question extraction & Natural question generation

Rule based approach
Found patterns:
- NP-SUJ / VN / NP-ATS: where NP-SUJ contains a single NER
    **Rules**:
    1. **Question**: *[Question word] VN NP-SUJ ?*; **Answer**: *NP-ATS*
        Good for sentence like: "[NP-SUJ | Arsène Lupin] [VP | est] [NP-ATS | un personnage de fiction français créé par Maurice Leblanc.]"
        -> Q: *Qui est Arsène Lupin ?*; A: *un personnage de fiction français créé par Maurice Leblanc*
    2. **Question**: *[Question word] VN NP-ATS ?*; **Answer**: *NP-SUJ*
        Good for sentence like: *[NP-SUJ| Le Brésil] [VP | est] [NP-ATS | le cinquième plus grand pays de la planète], ...*
        -> Q: *Quel est le cinquième plus grand pays de la planète ?*; A: *Le Brésil*
    
    **Question word choice rule**
    The *[Question word]* is determined by the label of the named entity in NP-SUJ or by the presence of words or group of words in NP-ATS.
    In order:
    1. If words like "le plus", "les moins", "les premiers", "la seconde", ... are present in NP-ATS -> use `Quel`.
    2. According to the named entity label
        Label | Question word
        ----- | ------------- 
        PER   | Qui
        LOC   | Où
        ORG   | Qu'est-ce que
        MISC  | Qu'est-ce que

#### Generated data

For now, on a small sub set of wiki dumps, we extracted **249** questions-response pairs using only the 2nd variant of the first identified patterns and only questions for which we use the question words *Quel* as they yield the best questions answers pairs qualitatively.

#### Problems

- Some NP-ATS don't capture enough of the sentence to make sense:
    ex: *En 1840, [NP-SUJ | Pierre-Joseph Proudhon] [VP | est] [NP-ATS | le premier] à se réclamer anarchiste ...*
    Question: *Quel est le premier ?*, Better Question: *Quel est le premier à se réclamer anarchiste ?*
- NP-SUJ NP-ATS implicit references:
    ex: *... [NP-SUJ| la commune d’Anvers] [VP| était] [NP-ATS | la plus peuplée de Belgique] ...*
    Question: *Quel était la plus peuplée de Belgique ?*, Better Question: *Quel était la commune la plus peuplée de Belgique ?*
- Further references: to the object of the context or the article:
- NP-SUJ or NP-ATS contains too much of the sentence:
    ex: *[NP-SUJ | L'Alberta] [VP | est] [VP-ATS | le plus grand producteur canadien de pétrole (l'Alberta possède la deuxième réserve mondiale de pétrole brut, derrière l'Arabie saoudite), de gaz naturel et de charbon].*
    Question: *Quel est le plus grand producteur canadien de pétrole (l'Alberta possède la deuxième réserve mondiale de pétrole brut, derrière l'Arabie saoudite), de gaz naturel et de charbon ?*; Better question : *Quel est le plus grand producteur canadien de pétrole ?* or *Quel est le plus grand producteur canadien de pétrole, de gaz naturel et de charbon ?*
- *Quel* should agree in gender and number

#### Solutions

- Better dumps should improve constituency parsing and the current question extraction performances
- Short references (NP-SUJ / NP-ATS references) could be spotted and resolved
- NP-SUJ and NP-ATS can be trimmed (ex remove all parenthesis)
- Agrement in gender and number can be easily derived the gender and numbers of the words "le plus", "les moins", "les premiers", "la seconde"
- Long reference cannot easily and accuratly be resolved 