CLI Usage
=========

For general usage and arguments description run ``uqa --help`` or ``uqa -h``.

Help is also avaible for subcommands (ex: ``uqa ner --help``)

General usage and arguments
---------------------------

``uqa`` CLI is a group of subcommands:

- clean
- ner
- constituency
- qas
- show
- validate
- split

General options allow to configure verbosity and logging behaviour:

.. list-table::
    :widths: 10 20 20 50
    :header-rows: 1

    *   - Alias
        - Option Name
        - Value(s)
        - Description
    *   - ``-h``
        - ``--help``
        -
        - Show help message
    *   - ``--quiet``
        - ``--silent``
        -
        - Disable console logging
    *   -
        - ``--no-log``
        -
        - Disable file logging
    *   - ``-v``
        - ``--verbosity``
        - | ``["debug", "info",``
          | ``"warning", "error"]``
        - | (default: info)
          | Set console logging level
    *   - ``-lf``
        - ``--log-file``
        - `path`
        - Log file path (default: `uqa.log`)
    *   - ``-lv``
        - ``--log-verbosity``
        - | ``["debug", "info",``
          | ``"warning", "error"]``
        - | (default: debug)
          | Set file logging level

``download`` subcommand
-----------------------

The ``download`` subcommand is a shortcut for downloading pre-trained model required to perform NER and constituency parsing.

Usage:::

    uqa [options] download [-n/--name STR] MODEL

``MODEL`` can either be `all`, `spacy` or `benepar`.

* `all` will install default models
* | `spacy` or `benepar` will install SpaCy french model or Benepar french constituency parsing model respectively
  | they can be used with ``-n / --name`` option to specify the model to install

See :doc:`models`

Data processing subcommands
---------------------------

Subcommands:

* clean
* ner
* constituency
* qas

Usage:::

    uqa [options] <subcommand> [options] SRC... DST

Data processing commands both read and write data, for each input file and output file is generated.

Positional arguments
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    *   - Argument
        - Value(s)
        - Description
    *   - SRC
        - path(s)
        - | Single path to a dir
          | one / mulitple path(s) to file(s)
    *   - DST
        - path
        - Single DST path

.. _data-reading-arguments:

Data reading
^^^^^^^^^^^^

.. list-table::
    :widths: 10 20 20 50
    :header-rows: 1

    *   - Alias
        - Option Name
        - Value(s)
        - Description
    *   - ``-h``
        - ``--help``
        -
        - Display help message
    *   - ``-d``
        - ``--dir``
        -
        - Indicate SRC as a directory
    *   - ``-if``
        - ``--input-format``
        - ``["json", "pickle"]``
        - | (default: "json")
          | File format to read
    *   - ``-df``
        - ``--data-format``
        - ``["default", "fquad"]``
        - | (default: "default")
          | Data format

A dataset can be stored as either `JSON` or `pickle` files.

``clean``, ``ner`` and ``constituency`` subcommands support reading data in either :ref:`default-data-format` or :ref:`fquad-data-format`
as they work independently from each others, wheras ``qas`` command depends on``ner`` and ``constituency`` steps ouputs
and thus only accepts `default` data format.

All subcommands can process:

* a single file
* mulitple files enumerated in SRC...
* a whole directory (with ``-d`` option)

For the directory case, all files and subdirectory are explored and files with the expected extension are processed
(``.json`` for json `input-format` and ``.pickle`` for a pickle `input-format`).

.. _data-writing-arguments:

Data writing
^^^^^^^^^^^^

.. list-table::
    :widths: 10 20 20 50
    :header-rows: 1

    *   - Alias
        - Option Name
        - Value(s)
        - Description
    *   - ``-h``
        - ``--help``
        -
        - Display help message
    *   - ``-O``
        - ``--override``
        -
        - | Override existing files
          | instead of skiping
    *   - ``-of``
        - ``--output-format``
        - ``["json", "pickle"]``
        - | (default: "json")
          | File format to write
    *   -
        - ``--json-indent``
        - ``int >= 0``
        - | (default: 0)
          | Json indentation when `-of` is `json`

| For ``clean``, ``ner``, and `` constituency`` the data-format of the written files is `default`.
| For ``qas`` it's `fquad`.

``DST`` positional parameter determine the path(s) of the written file(s):

.. list-table::
    :widths: 20 30 50
    :header-rows: 1

    *   - if SRC is ...
        - DST is ...
        - Notes
    *   - a single input file path
        - the single output file path
        -
    *   - multiple input file paths
        - | the path to the directory
          | where ouputs files are written
        - | Output files preserve their name
          | and are directly saved under DST
    *   - a single path to a dir
        - | the path to the directory
          | where ouputs files are written
        - | The internal structure of SRC dir is
          | preserved inside DST

For each input file, the output file path is generated before processing, if ``-o / --override`` is **not** set
and the output file path already exist then the file is **skipped**.

``validate`` subcommand
-----------------------

This command checks wheter FQuAD format file(s) containing question / answers are correct.

It checks:

* whether the file(s) structure is correct
* whether ``answers`` field values agree with the ``context`` field values

The command has the same arguments as described in :ref:`data-reading-arguments`
but will raise an error if ``-df / --data-format`` is set to `default`.

``split`` subcommand
--------------------

The ``split`` subcommand allow to split / combine data files.

Usage:::

    uqa split [options] NUM SRC... DST

The command has the same arguments as described in :ref:`data-reading-arguments`
and in :ref:`data-writing-arguments` but DST positional behaves differently.

The output files have the same data format as the input files.

if ``NUM`` is 0 or negative all SRC... files are combined and saved at DST.
if ``NUM`` is 1 or more, the DST must be a path template which contains a placeholder ``{num}``,
the input files data are read and split in chunks of ``NUM`` articles / documents.
The chunks are saved at DST by replacing ``{num}`` with increasing integers starting from `0`.

``show`` subcommand
-------------------

The ``show`` subcommand allow to visualize data with NER / constituency / qas information.

The command has the same arguments as described in :ref:`data-reading-arguments`.

And additonaly accepts:

.. list-table::
    :widths: 10 20 20 50
    :header-rows: 1

    *   - Alias
        - Option Name
        - Value(s)
        - Description
    *   - ``-a``
        - ``--all``
        -
        - Print for all contexts at once
    *   -
        - ``--no-ner``
        -
        - Discard NER information
    *   -
        - ``--no-const``
        -
        - Discard constituency information
    *   -
        - ``--depth``
        - int
        - | (default: -1)
          | Set maximum depth for constituents to be displayed
    *   -
        - ``--show-no-label``
        -
        - Display constituents with empty label.
    *   -
        - ``--rule``
        - ``['rule1', 'rule1_ext']``
        - If set, will apply the rule and display its outputs.

If ``-a / --all`` is not set the console will print one context at a time and prompt the user to continue.
