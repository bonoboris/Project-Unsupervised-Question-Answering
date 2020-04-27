"""Show command implementation."""

import logging

import click

from uqa import context_utils, dataset, qa_gen


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# class NavigableDataset:
#     """Navigable dataset explorer for show command (WIP)
#     Goals: Goto command / easy navigation.
#     """
#     def __init__(self, dataloader: dataset.DataLoader):
#         self.dataloader = dataloader
#         self.num_files = len(self.filepaths)

#         self.article_counts = []
#         logging.info("Computing dataset index.")
#         for fpath, fcontent in dataloader:
#             self.article_counts.append((fpath, len(fcontent)))
#         self.num_articles = sum(cnt for _, cnt in self.article_counts)

#         self.cur_article = 0

#         self._opened = (-1, ())

#     @property
#     def filepaths(self):
#         return self.dataloader.filepaths()

#     def __iter__(self):
#         while self.cur_article < self.num_articles:
#             txt = yield self._get_article(self.cur_article)
#             click.echo_via_pager(txt)
#             user_input = click.prompt("Goto article ? [default=next article]", default="", show_default=False)
#             if user_input:
#                 self.cur_article = int(user_input)
#             else:
#                 self.cur_article += 1

#     def _get_article(self, num_article: int):
#         if num_article >= self.num_articles or num_article < 0:
#             raise IndexError(f"Invalid `num_article` value {num_article}; must be between 0 and {self.num_articles}")

#         num_file, article_idx = self._to_file_article_nums(num_article)
#         fcontent = self._get_file_content(num_file)
#         return fcontent[article_idx]

#     def _to_file_article_nums(self, num_article):
#         for i, (_, article_count) in enumerate(self.article_counts):
#             if article_count < num_article:
#                 num_article -= article_count
#             else:
#                 return i, num_article
#         return -1, -1

#     def _get_file_content(self, num_file: int):
#         if num_file >= self.num_files or num_file < 0:
#             raise ValueError(f"Invalid `num_file` value {num_file}; must be between 0 and {self.num_files}")
#         if self._opened[0] != num_file:
#             self._opened = (
#                 num_file,
#                 self.dataloader._reader(self.filepaths[num_file]),  # pylint: disable=protected-access
#             )
#         return self._opened[1]


# def test_show_dl(dataloader: dataset.DataLoader):
#     nav_ds = NavigableDataset(dataloader)
#     nav_gen = iter(nav_ds)
#     for article in nav_gen:
#         l1 = f"Article num {nav_ds.cur_article} with id {article['id_article']} \nTitle: {article['title']}\n"

#         def _text_gen():
#             yield l1
#             for context in article["contexts"]:
#                 yield context["text"]

#         nav_gen.send(_text_gen)


def show_dl(
    data_it: dataset.DataIterable,
    depth: int,
    show_all: bool = False,
    no_ner: bool = False,
    no_const: bool = False,
    show_no_label: bool = False,
) -> None:
    """Show command implementation."""
    skip_val = ""
    context_it = context_utils.contextify(data_it)
    try:
        while True:
            if skip_val == "a":
                article_id = context.doc_id  # pylint: disable=used-before-assignment
                while context.doc_id == article_id:
                    context = next(context_it)
                skip_val = ""
            elif skip_val == "f":
                fpath = context.fpath  # pylint: disable=used-before-assignment
                while context.fpath == fpath:
                    context = next(context_it)
                skip_val = ""
            elif skip_val == "n":
                return
            else:
                context = next(context_it)

            click.echo(context.header(["white", "yellow"]))
            context.set_color_all("ner", "cyan")
            context.set_color_hier("constituents", ["magenta", "green", "blue", "red", "yellow"])
            labels = list()
            if not no_const:
                labels.extend([el.copy(depth=depth) for el in context.constituents])
            if not no_ner:
                labels.append(context.ner)
            for _, answer_label in context.qas:
                labels.append(answer_label.copy(color="cyan"))
            click.echo(context_utils.decorate(context.text, labels, show_no_label=show_no_label))
            for question, answer_label in context.qas:
                click.echo("\n")
                click.echo(context_utils.colorize(question, "cyan"))
                click.echo(context_utils.colorize(answer_label.extract(context.text), "white"))
            if not show_all:
                skip_val = click.prompt(
                    "Continue ? [(Y)es / (n)o / next (a)rticle/ next (f)ile]", default="", show_default=False
                )
    except StopIteration:
        return


RULE_MAP = {
    "rule1": qa_gen.rule1,
    "rule1_ext": qa_gen.rule1_ext,
}


def show_rule_dl(data_it: dataset.DataIterable, rule: str, show_all: bool = False) -> None:
    """Show command implementation for rules."""
    skip_val = ""
    context_it = context_utils.contextify(data_it)
    rule_fct = RULE_MAP[rule]
    try:
        while True:
            if skip_val == "a":
                article_id = context.doc_id  # pylint: disable=used-before-assignment
                while context.doc_id == article_id:
                    context = next(context_it)
                skip_val = ""
            elif skip_val == "f":
                fpath = context.fpath  # pylint: disable=used-before-assignment
                while context.fpath == fpath:
                    context = next(context_it)
                skip_val = ""
            elif skip_val == "n":
                return
            else:
                context = next(context_it)
            rule1_rt = rule_fct(context)
            while not rule1_rt:
                context = next(context_it)
                rule1_rt = rule_fct(context)

            qa_gen.rule1_to_qa(context, rule1_rt)
            click.echo(context.header(["white", "yellow"]))

            labels = list(rule1_rt)
            for _, answer_label in context.qas:
                labels.append(answer_label.copy(color="cyan"))
            click.echo(context_utils.decorate(context.text, labels))
            for question, answer_label in context.qas:
                click.echo("\n")
                click.echo(context_utils.colorize(question, "cyan"))
                click.echo(context_utils.colorize(answer_label.extract(context.text), "white"))
            if not show_all:
                skip_val = click.prompt(
                    "Continue ? [(Y)es / (n)o / next (a)rticle/ next (f)ile]", default="", show_default=False
                )
    except StopIteration:
        return
