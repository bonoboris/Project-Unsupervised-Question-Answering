from colorama import Fore
import math

def yellow(text):
    return Fore.YELLOW + text + Fore.RESET

def blue(text):
    return Fore.BLUE + text + Fore.RESET

def red(text):
    return Fore.RED + text + Fore.RESET

def colorize(text: str, color: str):
    """Colorize and return `text` with color `color`.
    
    Args
    ----
    color: str
        Has to be one of: black, red, green, yellow, blue, magenta, cyan, white
    """
    return getattr(Fore, color.upper()) + text + Fore.RESET

class Label(object):
    """Represent a span of text defined by its start and end indexes to decorate with a label and optionnaly a color and an offset.
    
    If orderable 
    """

    def __init__(self, start, end, label, color=None, offset=0):
        self.start = start
        self.end = end
        self.label = label
        self.color = color
        self.offset = offset
    
    def _check_order(self, **kwargs):
        off = kwargs.get("offset", getattr(self, "_offset", 0))
        start = kwargs.get("start", getattr(self, "_start", 0))
        end = kwargs.get("end", getattr(self, "_end" ,math.inf))
        if not (0 <= off and off <= start and start < end):
            raise ValueError("Attributes must verify: 0 <= `offset` <= `start` < `end`")

    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, val):
        self._check_order()
        self._offset = val
    
    @property
    def start(self):
        return self._start
    
    @start.setter
    def start(self, val):
        self._check_order()
        self._start = val

    @property
    def end(self):
        return self._end
    
    @end.setter
    def end(self, val):
        self._check_order()
        self._end = val

    def __lt__(self, other):
        if not isinstance(other, Label):
            return NotImplementedError
        if self.start != other.start:
            return self.start < other.start
        else:
            return self.end > other.end
    
    def __le__(self, other):
        if not isinstance(other, Label):
            return NotImplementedError
        if self.start != other.start:
            return self.start < other.start
        else:
            return self.end >= other.end 
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
         return not self < other
    
    def __eq__(self, other):
        if not isinstance(other, Label):
            return NotImplementedError
        return self.start == other.start and self.end == other.end and self.label == other.label

    def __neq__(self, other):
        return not self == other

    def __contains__(self, other):
        if not isinstance(other, Label):
            return NotImplementedError
        return self.start <= other.start and other.end <= self.end
    
    def __str__(self):
        s = f"{self.label} ({self.start}, {self.end})"
        if self.color:
            s += f" -> {self.color}"
        return s
    
    def decorate_span(self, span, template):
        if self.color:
            span = Fore.RESET + span + getattr(Fore, self.color.upper())
            template = colorize(template, self.color)
        return template.format(label=self.label, txt=span)

    def decorate(self, text, template):
        span = text[self.start: self.end]
        return self.decorate_span(span, template)


def decorate(text:str, labels:list, template:str ="[{label} {txt}]", autocoloring=False):
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
    if isinstance(labels[0], dict):
        labels = sorted((Label(**el) for el in labels))
    else:
        labels = sorted(labels)

    if autocoloring:
        colors = [el.strip().lower() for el in "red, green, yellow, blue, magenta, cyan".split(",")]
        for i, l in enumerate(labels):
            l.color = colors[i % len(colors)]

    def _rec(li, lj, ti, tj):
        # simple case: label interval == empty, return the text in [ti, tj]
        if lj == li:
            return text[ti:tj]
        # simple case: label interval == singleton, return the decorated text in [ti, tj]
        if lj-li == 1:
            label = labels[li]
            return text[ti : label.start] + labels[li].decorate(text, template) + text[label.end : tj]
        # rec:
        #   args are so that ti <= labels[li].start < labels[lj-1].end <= tj
        #   operations: inside labels[li:lj] and text[ti:tj]:
        #       - for each parent label find the final substring they decorate by rec (final meaning after decoration of childs labels)
        #           and apply the decoration
        #       - concatenate non decorated parts and parent decorated parts
        
        # add first non decorated part
        decorated = text[ti: labels[li].start]

        lp = li
        parent = labels[lp]
        for k in range(li+1, lj):
            if labels[k] not in parent:
                # get substring to be decorated
                substr = _rec(lp+1, k, parent.start, parent.end)
                # decorate and concatenate
                decorated += parent.decorate_span(substr, template)
                # concatenate non decorated part
                decorated += text[parent.end: labels[k].start]
                # update parent, lp
                lp = k
                parent = labels[lp]
        # add last parent
        substr = _rec(lp+1, lj, parent.start, parent.end)
        decorated += parent.decorate_span(substr, template)

        # add last non decorated part
        decorated += text[parent.end: tj]
        return decorated

    return _rec(0, len(labels), 0, len(text))

            


if __name__ == "__main__":
    txt = "".join(map(chr, range(65, 65 + 26)))
    print(txt)
    # labels = [
    #     Label(2, 10, "1", "blue"),  # CDEFGHIJ
    #     Label(3, 7, "2", "red"),  # DEFG
    #     Label(5, 7, "3", "yellow"), # FG
    #     Label(14, 26, "4", "cyan"), # OPQRSTUVWXYZ
    #     Label(14, 18, "5", "green"), # OPQRST
    #     Label(20, 23, "6", "magenta"), # UVW
    #     Label(23, 26, "7", "blue"), # XYZ
    # ]
    labels = [
        Label(2, 10, "1"),  # CDEFGHIJ
        Label(3, 7, "2"),  # DEFG
        Label(5, 7, "3"), # FG
        Label(14, 26, "4"), # OPQRSTUVWXYZ
        Label(14, 18, "5"), # OPQRST
        Label(20, 23, "6"), # UVW
        Label(23, 26, "7"), # XYZ
    ]
    print(decorate(txt, labels))