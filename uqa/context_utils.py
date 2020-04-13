"""Imporved context data structures."""

import copy as cp
import math
import random as rd
from collections import OrderedDict
from os import path
from typing import Any, Dict, Generator, Iterable, List, NamedTuple, Optional, Text, Tuple, Union

from colorama import Fore


class SimpleRepr(object):
    """A mixin implementing a simple __repr__."""

    def __repr__(self):
        return "{klass}({attrs})".format(
            klass=self.__class__.__name__, attrs=", ".join("{}={!r}".format(k, v) for k, v in vars(self).items()),
        )


def yellow(text: str) -> str:
    return Fore.YELLOW + text + Fore.RESET


def blue(text: str) -> str:
    return Fore.BLUE + text + Fore.RESET


def red(text: str) -> str:
    return Fore.RED + text + Fore.RESET


def colorize(text: str, color: str):
    """Colorize and return `text` with color `color`.
    
    Args
    ----
    color: str
        Has to be one of: black, red, green, yellow, blue, magenta, cyan, white
    """
    return getattr(Fore, color.upper()) + text + Fore.RESET


JsonT = Any


class Label(SimpleRepr):
    """Represent a span of text defined by its start and end indexes to decorate with a label and optionnaly a color and an offset.
    
    Implements order operators so that sorting a list of labels returns label with increasing start indexes and from the bigger to the smaller for
    labels sharing the same start index

    Also implement equality operator, two labels are equal if the have the same start, end indices and label. 

    Finaly implements the `in` operator which returns True for label (strictly or not) included in another one. 

    Attributes
    ----------
        start: int
            start index of the label
        end: int
            end index of the label
        label: string
            label category
        extras: Dict[str, Any]
            holds extra informations such as color or printing offset when pretty printing.

    """

    def __init__(self, start: int, end: int, label: str, **kwargs: Any):
        self.start = start
        self.end = end
        self.label = label
        self.extras: Dict[str, Any] = dict(**kwargs)

    def extract(self, text: str) -> str:
        off = self.extras.get("offset", 0)
        return text[off + self.start : off + self.end]

    def to_json(self, exclude_extras: bool = True) -> JsonT:
        ret = vars(self)
        if exclude_extras:
            if "extras" in ret:
                del ret["extras"]
        return ret

    def __getitem__(self, key: str) -> Any:
        """Get instance attibrute."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set instance attribute."""
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
        """Returns a copy of the label.
        
        Args
        ----
            extras: **Any
                entry to add / update in the copy `extras` attribute.
        """
        cop = cp.copy(self)
        for k, v in extras.items():
            cop.extras[k] = v
        return cop


class LabelNode(Label):
    """Extend Label with a `children` attribute to keep track of labels hierarchy."""

    def __init__(self, start, end, label, children=list(), **kwargs):
        super().__init__(start, end, label, **kwargs)
        self.children: List["LabelNode"] = list(children)

    def to_json(self, exclude_extras: bool = True) -> JsonT:
        """Returns the hierachy as a json dumpable structure"""
        ret = super().to_json(exclude_extras)
        for i in range(len(ret["children"])):
            ret["children"][i] = ret["children"][i].to_json()
        ord_ret = OrderedDict()
        ord_ret["label"] = ret["label"]
        ord_ret["start"] = ret["start"]
        ord_ret["end"] = ret["end"]
        other_keys = set(ret).difference({"label", "start", "end", "children"})
        ord_ret.update({k: ret[k] for k in other_keys})
        ord_ret["children"] = ret["children"]
        return ord_ret

    @classmethod
    def from_json(cls, jsonlike: JsonT) -> "LabelNode":
        """Instaciate and return an instance from a json like data structure."""
        children = jsonlike.pop("children")
        inst = cls(**jsonlike)
        inst.children = list()
        for child in children:
            inst.children.append(cls.from_json(child))
        return inst

    def flat_iter(self) -> Generator["LabelNode", None, None]:
        """Iterate the sub tree starting from this node from lowest to biggest (relative to Label order).
        
        The first label yielded is always self.
        """
        yield self
        for c in self.children:
            yield from c.flat_iter()

    def to_label(self) -> Label:
        """Returns a Label instance containing the same attributes except for `children`"""
        return Label(self.start, self.end, self.label, **self.extras)

    def copy(self, depth=-1):
        """Copy this instance with limited depth.
        
        Args
        ----
            depth (default: -1): int 
                The depth to which the copy is done, if negative the entire hierarchy is copied, if 0, similar to self.copy_no_child()
        """
        cop = self.copy_no_child()
        if depth != 0:
            for child in self.children:
                cop.children.append(child.copy(depth - 1))
        return cop

    def copy_no_child(self, **extras: Any) -> "LabelNode":
        """Copy this instance without any child, any keyword argument add / update entries in copy `extras` attribute."""
        cop = cp.deepcopy(self)
        cop.children = list()
        for k, v in extras.items():
            cop.extras[k] = v
        return cop

    @classmethod
    def from_labels(cls, labels: Iterable[Label]) -> List["LabelNode"]:
        """Build labels hierachy and return it as a list of LabelNode."""
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
    """Set the color `color` to all labels in `label_iterable`, if the iterable contains a label node, set the color to the whole hierarchy"""
    for el in label_iterable:
        if isinstance(el, LabelNode):
            el.extras["color"] = color
            for sub_label in el.flat_iter():
                sub_label.extras["color"] = color
        elif isinstance(el, Label):
            el.extras["color"] = color
        else:
            set_color_all(el, color)


def set_color_hier(
    label_nodes: Union[LabelNode, Iterable[LabelNode]], colors: Optional[List[str]] = None, depth_off: int = 0
) -> None:
    """Cycle through colors and assign color to labels in label_nodes and their hierarchy according to their depth.
    
    Args
    ----
        label_node: Union[LabelNode, Iterable[LabelNode]]
            a single LabelNode or an iterable of LabelNode
        colors:
            List of colors to use
        depth_off:
            The depth offset of the label(s) in label_nodes, the first color used is colors[depth_off % len(colors)].
    """
    if colors is None:
        colors = [el.strip().lower() for el in "red, green, yellow, blue, magenta, cyan".split(",")]
    if isinstance(label_nodes, LabelNode):
        label_nodes.extras["color"] = colors[depth_off % len(colors)]
        set_color_hier(label_nodes.children, colors, depth_off + 1)
    else:
        for el in label_nodes:
            set_color_hier(el, colors, depth_off)


class QA(NamedTuple, SimpleRepr):
    """A question / answer pair of labels, the question is a string while the answer is a Label"""

    question: str
    answer: Label

    def to_json(self, exclude_extras: bool = True) -> Dict[str, Any]:
        return OrderedDict(question=self.question, answer=self.answer.to_json(exclude_extras))


class Context(SimpleRepr):
    """Represent a context with its information.
    
    Attributes
    ----------
        fpath: The file from which this context was red from
        doc_id: The context's document (article) id
        doc_title: The context's document title
        context_id: The context's index in its document
        text: The context raw text
        ner: A list of Label with one label per named entity.
        constituents: A list of LabelNode, with one LabelNode per sentence.   
    """

    def __init__(
        self,
        fpath: str,
        doc_id: int,
        doc_title: str,
        context_id: int,
        text: str,
        ner: List[Label] = list(),
        constituents: List[LabelNode] = list(),
        pre_cloze: List[QA] = list(),
        qas: List[QA] = list(),
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
    def from_json(cls, data: JsonT) -> "Context":
        """Instanciate a context from jsonlike data structures with keys corresponding to this class attributes names."""
        inst: Context = cls(data["fpath"], data["doc_id"], data["doc_title"], data["context_id"], data["text"])
        if "ner" in data:
            for ent in data["ner"]:
                inst.ner.append(Label(**ent))
        if "constituents" in data:
            for sent_consts in data["constituents"]:
                inst.constituents.append(LabelNode.from_json(sent_consts))
        if "qas" in data:
            for qa in data["qas"]:
                inst.qas.append(QA(question=qa["question"], answer=qa["answer"]))
        return inst

    def set_color_all(self, attr_name: str, color: str) -> None:
        set_color_all(getattr(self, attr_name), color)

    def set_color_hier(self, attr_name: str, colors: Optional[List[str]] = None) -> None:
        set_color_hier(getattr(self, attr_name), colors)

    def header(self, color: Optional[Union[str, Tuple[str, str]]] = None) -> str:
        """Returns a string with the file path in the first line and the article id, title and the context id in the second line."""
        l1 = self.fpath + "\n"
        l2 = f"{self.doc_id} - {self.doc_title} [{self.context_id}]\n"
        if color is None:
            return l1 + l2
        elif isinstance(color, Text):
            return colorize(l1 + l2, color)
        else:
            col1, col2 = color
            return colorize(l1, col1) + colorize(l2, col2)


def decorate_span(text_span: str, label: Label, template: str) -> str:
    """Decorate the whole text_span with label following the template. (label.start and label.end are ignored)"""
    if "color" in label.extras:
        color: str = label.extras["color"]
        text_span = Fore.RESET + text_span + getattr(Fore, color.upper())
        template = colorize(template, color)
    return template.format(label=label.label, txt=text_span)


def decorate(text: str, labels: list, template: str = "[{label} {txt}]", autocoloring=False) -> str:
    """Decorate sub-string in `text` according to indexes and labels in labels.
    
    Args
    ----
    text: str
        The text to decorate
    labels: list
        List of dictionnary containing the items "start":int, "end":int, "label":str and optionnaly "color":str.
        Notes:
            - "start" and "end" must be valid indexes in text.
            - if "color" is set it must be one of black, red, green, yellow, blue, magenta, cyan, white;
                the decoration string defined by format will be colorized exect for the `txt` part.
    format: str
        template for the decoration part must contains placeholders {label} and {txt}.

    Notes
    -----
        labels can contain nested entries but must not contain "crossing" entries.
    """

    def flatten(obj):
        ret = list()
        for el in obj:
            if isinstance(el, LabelNode):
                ret.extend(el.flat_iter())
            elif isinstance(el, Label):
                ret.append(el)
            elif isinstance(el, dict):
                ret.append(Label(**el))
            else:
                ret.extend(flatten(el))
        return ret

    labels = flatten(labels)
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


def contextify(json_file_it: Iterable[Tuple[str, JsonT]]) -> Generator[Context, None, None]:
    jcontext = dict()
    for fpath, json_content in json_file_it:
        jcontext["fpath"] = fpath
        for article in json_content:
            jcontext["doc_id"] = article["id_article"]
            jcontext["doc_title"] = article["title"]
            for para in article["contexts"]:
                jcontext["context_id"] = para["id_context"]
                jcontext["text"] = para["text"]
                jcontext["ner"] = para.get("entities", ())
                jcontext["constituents"] = para.get("constituency", ())
                yield Context.from_json(jcontext)


def contextify_rd(json_file_it) -> Generator[Context, None, None]:
    for el in json_file_it:
        fcontexts = list(contextify((el,)))
        rd.shuffle(fcontexts)
        for context in fcontexts:
            yield context


def jsonify(context_it: Iterable[Context], include_qas: bool = True) -> Iterable[Tuple[str, JsonT]]:
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
