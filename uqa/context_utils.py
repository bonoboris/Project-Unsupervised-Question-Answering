"""Data classes.

| :class:`Context` class stores context's raw text, processing steps outputs and meta data.
| :class:`Label` class stores any extra information about a span of text.
| :class:`LabelNode` class extends :class:`Label` to represent hierarchical information on a raw text
  such as Constituency parsing results.

The classes implement features for easy manipulation, visualization and IO operations.
"""

import collections
import copy as cp
import math
import random as rd
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, NamedTuple, Optional, Text, Tuple, Union

from colorama import Fore

from uqa import dataset


class _SimpleRepr(object):
    """A mixin implementing a simple __repr__."""

    def __repr__(self):
        return "{klass}({attrs})".format(
            klass=self.__class__.__name__, attrs=", ".join("{}={!r}".format(k, v) for k, v in vars(self).items()),
        )


def yellow(text: str) -> str:
    """Enclose and return `text` str with ASCII yellow color espace sequence."""
    return Fore.YELLOW + text + Fore.RESET


def blue(text: str) -> str:
    """Enclose and return `text` str with ASCII blue color espace sequence."""
    return Fore.BLUE + text + Fore.RESET


def red(text: str) -> str:
    """Enclose and return `text` str with ASCII red color espace sequence."""
    return Fore.RED + text + Fore.RESET


def colorize(text: str, color: str):
    """Colorize and return `text` with color `color`.

    Use ASCII color escape sequence and will only display properly on terminal supporting those espace sequence.

    Args
    ----
    color: str
        Has to be one of: black, red, green, yellow, blue, magenta, cyan, white
    """
    return getattr(Fore, color.upper()) + text + Fore.RESET


class Label(_SimpleRepr):
    """Represent label span of text defined by its start and end indices and a label string.

    `Label` instances also have an :attr:`extras` attribute to hold any other characteristic,
    such as `color` for printing purpose.

    Implements order operators so that sorting a list of labels returns label
    with increasing :attr:`start` indices and from the outermost label to inner most label.

    Two labels are equals if they have the same :attr:`start`, :attr:`end` indices and :attr:`label`.

    The `in` operator returns ``True`` is the left operand label is (strictly or not) included
    in the right operand label.

    Also implement dictionnary like attribute accessor for convinience.

    Attributes
    ----------
    start: int
        start index of the label
    end: int
        end index of the label
    label: string
        label category
    extras: Dict[str, Any]
        holds extra informations such as color for pretty printing.
    """

    def __init__(self, start: int, end: int, label: str, **extras: Any):
        """
        Parameters
        ----------
        start: int
            start index of the label
        end: int
            end index of the label
        label: string
            label category

        Keyword Arguments
        -----------------
        extras:
            Stored as :attr:`extras` entries
        """
        self.start = start
        self.end = end
        self.label = label
        self.extras: Dict[str, Any] = dict(**extras)

    def extract(self, text: str) -> str:
        """Return text[:attr:`.start`, :attr:`.end`]"""
        off = self.extras.get("offset", 0)
        return text[off + self.start : off + self.end]

    def to_json(self, exclude_extras: bool = True) -> Dict:
        """Returns the instance attributes as a json-like dict.

        Parameters
        ----------
        exclude_extras: bool, default=True
            If True, exclude :attr:`.extras` value from the result.
        """
        ret = vars(self)
        if exclude_extras:
            if "extras" in ret:
                del ret["extras"]
        return ret

    def __getitem__(self, key: str) -> Any:
        """Get instance attibrute with name `key`."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set instance attribute `key` with `value`."""
        return setattr(self, key, value)

    def __lt__(self, other: "Label") -> bool:
        if not isinstance(other, Label):
            return NotImplementedError
        if self.start != other.start:
            return self.start < other.start
        else:
            return self.end > other.end

    def __le__(self, other: "Label") -> bool:
        if not isinstance(other, Label):
            return NotImplementedError
        if self.start != other.start:
            return self.start < other.start
        else:
            return self.end >= other.end

    def __gt__(self, other: "Label") -> bool:
        return not self <= other

    def __ge__(self, other: "Label") -> bool:
        return not self < other

    def __eq__(self, other: "Label") -> bool:
        if not isinstance(other, Label):
            return NotImplementedError
        return self.start == other.start and self.end == other.end and self.label == other.label

    def __neq__(self, other: "Label") -> bool:
        return not self == other

    def __contains__(self, other: "Label") -> bool:
        if not isinstance(other, Label):
            return NotImplementedError
        return self.start <= other.start and other.end <= self.end

    def copy(self, **extras: Any):
        """Returns a deep copy of the label.

        Keyword Arguments
        ------------------
        extras
            Entries to add / update in the copied :attr:`extras` attribute.
        """
        cop = cp.deepcopy(self)
        for key, val in extras.items():
            cop.extras[key] = val
        return cop


class LabelNode(Label):
    """Extend :class:`Label` with a :attr:`children` attribute to represent a labels hierarchy.

    Attributes
    ----------
    children: list of :class:`LabelNode`
        List of childen in ascending order
    """

    def __init__(self, start: int, end: int, label: str, children: List["LabelNode"] = (), **extras: Any):
        """
        Parameters
        ----------
        start: int
            start index of the label
        end: int
            end index of the label
        label: string
            label category
        children: Iterable[LabelNode]
            List of children nodes.

        Keyword Arguments
        -----------------
        extras:
            Stored as :attr:`.extras` entries
        """
        super().__init__(start, end, label, **extras)
        self.children: List["LabelNode"] = list(children)

    def to_json(self, exclude_extras: bool = True) -> Dict:
        """Returns the hierachy labels hierarchy as a json-like dict.

        Parameters
        ----------
        exclude_extras: bool, default=True
            If True, exclude :attr:`.extras` value from the result.
        """
        ret = super().to_json(exclude_extras)
        for i in range(len(ret["children"])):
            ret["children"][i] = ret["children"][i].to_json()
        ord_ret = collections.OrderedDict()
        ord_ret["label"] = ret["label"]
        ord_ret["start"] = ret["start"]
        ord_ret["end"] = ret["end"]
        other_keys = set(ret).difference({"label", "start", "end", "children"})
        ord_ret.update({k: ret[k] for k in other_keys})
        ord_ret["children"] = ret["children"]
        return ord_ret

    @classmethod
    def from_json(cls, jsonlike: MutableMapping) -> "LabelNode":
        """Instaciate and return an `LabelNode` instance from a json-like data structure.

        `jsonlike` keys name must match the attribute names,
        any extra field content will be stored in :attr:`.extras`.
        """
        children = jsonlike.pop("children", list())
        inst = cls(**jsonlike)
        inst.children = list()
        for child in children:
            inst.children.append(cls.from_json(child))
        return inst

    def flat_iter(self) -> Iterable["LabelNode"]:
        """Iterate the sub tree defined by this node from lowest to biggest (relative to Label order).

        The first label yielded is always the instance bounded.
        """
        yield self
        for child in self.children:
            yield from child.flat_iter()

    def to_label(self) -> Label:
        """Returns a Label instance containing the same :attr:`.start`, :attr:`.end` and :attr:`.extras` values."""
        return Label(self.start, self.end, self.label, **self.extras)

    def copy(self, depth=-1):  # pylint: disable=arguments-differ
        """Return a deep-copy of this instance.

        Parameters
        ----------
        depth: int, default=-1
            The depth to which the copy is done, if negative the entire hierarchy is copied,
            if 0, similar to self.copy_no_child()

        Returns
        -------
        :class:`LabelNode`
            The new copied instance.
        """
        cop = self.copy_no_child()
        if depth != 0:
            for child in self.children:
                cop.children.append(child.copy(depth - 1))
        return cop

    def copy_no_child(self, **extras: Any) -> "LabelNode":
        """Copy this instance without copying its children.

        Keyword Arguments
        -----------------
        extras:
            Udpate the copied instance :attr:`.extras` dictionnary
        """
        cop = cp.deepcopy(self)
        cop.children = list()
        cop.extras.update(extras)
        return cop

    @classmethod
    def from_labels(cls, labels: Iterable[Label]) -> List["LabelNode"]:
        """Build and return a list LabelNode hierarchies from `labels` iterable.

        Parameters
        ----------
        labels: Iterable of :class:`Label`
            An iterable of labels either included in ou distinct from one another.

        Returns
        -------
        list of :class:`LabelNode`
            The forest (graph theory) made of labels in `labels` iterable as a list of the roots of each tree.
        """
        labels: Iterable[Label] = sorted(labels)
        root = cls(0, math.inf, label="")
        parents: List["LabelNode"] = [root]
        for l in labels:
            while l not in parents[-1]:
                parents.pop()
            inst = cls(**vars(l))
            parents[-1].children.append(inst)
            parents.append(inst)
        return root.children


def set_color_all(label_iterable: Iterable["Label"], color: str) -> None:
    """Set the color `color` to all labels in `label_iterable`, if the iterable contains a label node
    set the color to the whole hierarchy"""
    for label in label_iterable:
        if isinstance(label, LabelNode):
            label.extras["color"] = color
            for sub_label in label.flat_iter():
                sub_label.extras["color"] = color
        elif isinstance(label, Label):
            label.extras["color"] = color
        else:
            set_color_all(label, color)


def set_color_hier(
    label_nodes: Union[LabelNode, Iterable[LabelNode]], colors: Optional[List[str]] = None, depth_offset: int = 0
) -> None:
    """Cycle through `colors` and assign a color to each labels in `label_nodes`
    and their hierarchy depending to their depth.

    Parameters
    ----------
    label_node: :class:`LabelNode` or Iterable of :class:`LabelNode`
        a single :class:`LabelNode` or an iterable of :class:`LabelNode`
    colors:
        List of colors to use
    depth_offset:
        The depth offset of the label(s) in `label_nodes`, the first color used is colors[depth_off % len(colors)].
    """
    if colors is None:
        colors = [color.strip().lower() for color in "red, green, yellow, blue, magenta, cyan".split(",")]
    if isinstance(label_nodes, LabelNode):
        label_nodes.extras["color"] = colors[depth_offset % len(colors)]
        set_color_hier(label_nodes.children, colors, depth_offset + 1)
    else:
        for label in label_nodes:
            set_color_hier(label, colors, depth_offset)


class QA(NamedTuple, _SimpleRepr):
    """A question / answer pair of labels, the question is a string while the answer is a :class:`Label`"""

    question: str
    answer: Label

    def to_json(self, exclude_extras: bool = True) -> Dict[str, Any]:
        """Return a json-like dict representing the :class:`QA` instance."""
        return collections.OrderedDict(question=self.question, answer=self.answer.to_json(exclude_extras))


class Context(_SimpleRepr):
    """Represent a context and its computed features.

    Attributes
    ----------
    fpath: str
        The file from which this context was red from
    doc_id: int
        The context's document (article) id
    doc_title: str
        The context's document title
    context_id: int
        The context's index in its document
    text: str
        The context raw text
    ner: Iterable of :class:`Label`
        The list of computed named entities.
    constituents: Iterable of :class:`LabelNode`
        The forest of constituent as a list of :class:`LabelNode`; each root is a sentence in the :attr:`text`.
    qas: Iterable of :class:`QA`
        The list of generated question answer pairs
    """

    def __init__(
        self,
        fpath: str,
        doc_id: int,
        doc_title: str,
        context_id: int,
        text: str,
        ner: Iterable[Label] = (),
        constituents: Iterable[LabelNode] = (),
        qas: Iterable[QA] = (),
    ):
        self.fpath: str = fpath
        self.doc_id: int = doc_id
        self.doc_title: str = doc_title
        self.context_id: int = context_id
        self.text: str = text
        self.ner: List[Label] = list(ner)
        self.constituents: List[LabelNode] = list(constituents)
        self.qas: List[QA] = list(qas)

    @classmethod
    def from_json(cls, data: Dict) -> "Context":
        """Instanciate a context from json-like data structure with keys matching this class attributes names.

        The classmethod cast sub-dictionnaries of `data` into the appropriate types :class:`Label`, :class:`LabelNode`
        and :class:`QA`.
        """
        inst: Context = cls(data["fpath"], data["doc_id"], data["doc_title"], data["context_id"], data["text"])
        if "ner" in data:
            for ent in data["ner"]:
                inst.ner.append(Label(**ent))
        if "constituents" in data:
            for sent_consts in data["constituents"]:
                inst.constituents.append(LabelNode.from_json(sent_consts))
        if "qas" in data:
            for qa in data["qas"]:
                inst.qas.append(QA(question=qa["question"], answer=Label(**qa["answer"])))
        return inst

    def set_color_all(self, attr_name: str, color: str) -> None:
        """Calls :func:`set_color_all` on the attribute named `attr_name` with `color`."""
        set_color_all(getattr(self, attr_name), color)

    def set_color_hier(self, attr_name: str, colors: Optional[List[str]] = None) -> None:
        """Calls :func:`set_color_hier` on the attribute named `attr_name` with `colors`."""
        set_color_hier(getattr(self, attr_name), colors)

    def header(self, color: Optional[Union[str, Tuple[str, str]]] = None) -> str:
        """Returns a string with the file path in the first line and the article id,
        title and the context id in the second line.

        Parameters
        ----------
        color: None, str, or pair of str, default=None
            | If None, no coloring is applied,
            | if one color is passed, colorize both lines with it
            | if a pair of colors is passed colorize each line with one the colors.
        """
        line1 = self.fpath + "\n"
        line2 = f"{self.doc_id} - {self.doc_title} [{self.context_id}]"
        if color is None:
            return line1 + line2
        elif isinstance(color, Text):
            return colorize(line1 + line2, color)
        else:
            col1, col2 = color
            return colorize(line1, col1) + colorize(line2, col2)


def decorate_span(text_span: str, label: Label, template: str = "[{label} {txt}]") -> str:
    """Decorate the whole `text_span` with `label` following the `template`.

    | If `label` contains a `color` entry in its :attr:`Label.extras` attribute, the color is applied
      to all characters but the `text_span`.
    | :attr:`Label.start` and :attr:`Label.end` are ignored.

    Parameters
    ----------
    text_span: str
        Piece of text to decorate
    label: :class:`Label`
        A :class:`Label` or :class:`LabelNode` instance
    template: str, default="[{label} {txt}]"
        A template str with placeholders `{label}` and `{txt}`

    Returns
    -------
    str:
        `text_span` decorated with `label` according to the template
    """
    if "color" in label.extras:
        color: str = label.extras["color"]
        text_span = Fore.RESET + text_span + getattr(Fore, color.upper())
        template = colorize(template, color)
    return template.format(label=label.label, txt=text_span)


def decorate(
    text: str, labels: Iterable, template: str = "[{label} {txt}]", autocoloring=False, show_no_label=False
) -> str:
    """Decorate `text` with `labels` according to the `template`.

    Parameters
    ----------
    text: str
        The text to decorate
    labels: Iterable
        An recursivesly iterable container where all leaves are either :class:`Label`, :class:`LabelNode` or
        dict representing a Label.
    format: str
        template for the decoration part must contains placeholders {label} and {txt}.
    autocoloring: bool, default=False
        If ``True`` override `color` :attr:`.extras` setting with cyclic coloring of the label.
    show_no_label: bool, default=False
        If ``False`` don't use `labels` with empty :attr:`~.Label.label` value.

    Returns
    -------
    str:
        The decorated colorized string

    Notes
    -----
        `labels` must be either nested in or distinct from one another.
    """

    def flatten(labels_it: Iterable):
        ret = list()
        for obj in labels_it:
            if isinstance(obj, LabelNode):
                ret.extend(obj.flat_iter())
            elif isinstance(obj, Label):
                ret.append(obj)
            elif isinstance(obj, Mapping):
                ret.append(Label(**obj))
            else:
                ret.extend(flatten(obj))
        return ret

    labels = flatten(labels)
    if not show_no_label:
        labels = [l for l in labels if l.label]
    labels = sorted(labels)

    if autocoloring:
        colors = [el.strip().lower() for el in "red, green, yellow, blue, magenta, cyan".split(",")]
        for i, l in enumerate(labels):
            l.extras["color"] = colors[i % len(colors)]

    def _rec(li, lj, ti, tj):
        # simple case: label interval == empty, return the text in [ti, tj]
        if lj == li:
            return text[ti:tj]
        # simple case: label interval == singleton, return the decorated text in [ti, tj]
        if lj - li == 1:
            label = labels[li]
            _before = text[ti : label.start]
            _to_dec = text[label.start : label.end]
            _after = text[label.end : tj]
            return _before + decorate_span(_to_dec, label, template) + _after
        # rec:
        #   args are so that ti <= labels[li].start < labels[lj-1].end <= tj
        #   operations: inside labels[li:lj] and text[ti:tj]:
        #       - for each parent label find the final substring they decorate by rec (final meaning after decoration of childs labels)
        #           and apply the decoration
        #       - concatenate non decorated parts and parent decorated parts

        # add first non decorated part
        decorated = text[ti : labels[li].start]

        lp = li
        parent = labels[lp]
        for k in range(li + 1, lj):
            if labels[k] not in parent:
                # get substring to be decorated
                substr = _rec(lp + 1, k, parent.start, parent.end)
                # decorate and concatenate
                decorated += decorate_span(substr, parent, template)
                # concatenate non decorated part
                decorated += text[parent.end : labels[k].start]
                # update parent, lp
                lp = k
                parent = labels[lp]
        # add last parent
        substr = _rec(lp + 1, lj, parent.start, parent.end)
        decorated += decorate_span(substr, parent, template)

        # add last non decorated part
        decorated += text[parent.end : tj]
        return decorated

    return _rec(0, len(labels), 0, len(text))


def contextify(data_it: dataset.DataIterable) -> Iterable[Context]:
    """Extract and yield :class:`Context` instances from `default` structre iterable `data_it`."""
    jcontext = dict()
    for fpath, fcontent in data_it:
        jcontext["fpath"] = fpath
        for article in fcontent:
            jcontext["doc_id"] = article["id_article"]
            jcontext["doc_title"] = article["title"]
            for para in article["contexts"]:
                jcontext["context_id"] = para["id_context"]
                jcontext["text"] = para["text"]
                jcontext["ner"] = para.get("entities", ())
                jcontext["constituents"] = para.get("constituency", ())
                jcontext["qas"] = para.get("qas", ())
                yield Context.from_json(jcontext)


def contextify_rd(data_it) -> Iterable[Context]:
    """Extract and shuffle and yield :class:`Context` instances from `default` structre iterable `data_it`.

    Shuffling is done per file content.
    """
    for fpath, fcontent in data_it:
        fcontexts = list(contextify((fpath, fcontent)))
        rd.shuffle(fcontexts)
        for context in fcontexts:
            yield context


def jsonify(context_it: Iterable[Context], include_qas: bool = True) -> dataset.DataIterable:
    """Reverse operation of :func:`conetxtify`,
    recrate a dataset iterable from an ordered `context_it`.

    `context_it` must yield context in the order it was created by :func:`contextify`.

    Parameters
    ----------
    context_it: Iterable of :class:`Context`
        Context iterable
    include_qas: bool, default=True
        If ``False`` discard :attr:`Context.qas` values.
    """
    articles_dict = dict()
    cur_fpath = None
    for context in context_it:
        if cur_fpath is None:
            cur_fpath = context.fpath
        if context.fpath != cur_fpath:
            json_content = [article for _, article in sorted(articles_dict.items())]
            yield cur_fpath, json_content
            articles_dict.clear()
            cur_fpath = context.fpath
        else:
            if context.doc_id not in articles_dict:
                articles_dict[context.doc_id] = dict(id_doc=context.doc_id, title=context.doc_title, contexts=list())
            dct = dict(
                id_context=context.context_id,
                text=context.text,
                entities=[el.to_json() for el in context.ner],
                constituency=[el.to_json() for el in context.constituents],
            )
            if include_qas:
                dct["qas"] = [qa.to_json() for qa in context.qas]
            articles_dict[context.doc_id]["contexts"].append(dct)

    # yield last json file content
    json_content = [article for _, article in sorted(articles_dict.items())]
    yield cur_fpath, json_content
