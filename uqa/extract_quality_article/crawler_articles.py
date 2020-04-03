from bs4 import BeautifulSoup
import urllib3
import re
import pickle
import time
from uqa.extract_quality_article.quality_article_url_crawler import url_crawler

"""
    This file crawls into the different wikipedia urls and get the different paragraphs from them, storing them
    as contexts.
    
    {
        'id_article': 0,
        'title': 'Title of the article',
        contexts: []
    }
"""

base_url_wikipedia = 'https://fr.wikipedia.org'


def get_url_list():
    try:
        url_list = []
        url_database_txt = 'urls_articles_qualites.txt'
        with open(url_database_txt, 'r') as file:
            for line in file.readlines():
                url_list.append(line.rsplit('\n')[0])
    except:
        url_list = url_crawler()
    return url_list


def find_best_size_for_context(total_size, delimiter):
    nb_parts = 2
    while total_size / nb_parts > delimiter:
        nb_parts += 1
    return int(total_size / nb_parts)


def context_delimiter(paragraph, delimiter):
    if len(paragraph) < delimiter:
        return 1, [paragraph]

    # Split the text by phrase
    phrase_list = paragraph.split('.')
    phrase_list.pop()
    contexts = []
    current_phrase = ''
    max_size = find_best_size_for_context(len(paragraph), delimiter)
    for phrase in phrase_list:
        possible_phrase = current_phrase + phrase + '.'
        if len(current_phrase) > max_size and current_phrase != '':
            contexts.append(current_phrase)
            current_phrase = phrase + '.'
        else:
            current_phrase = possible_phrase
    contexts.append(current_phrase)
    return len(contexts), contexts


def wikipedia_crawler(base_url):
    articles = []
    url_list = get_url_list()
    nb_articles = len(url_list)
    http = urllib3.PoolManager()
    for article_id, url in enumerate(url_list):
        article = {'id_article': article_id}
        wiki_page = ''
        while wiki_page == '':
            try:
                wiki_page = http.request('GET', base_url + url).data.decode('utf-8')
                break
            except:
                print("Error with Server")
                time.sleep(5)
        soup = BeautifulSoup(wiki_page, 'html.parser')
        full_article_text = soup.find('div', class_="mw-parser-output")
        article['title'] = soup.find('h1', class_="firstHeading").text
        raw_paragraphs = full_article_text.find_all(['p', 'ul'], recursive=False)
        contexts = []
        id_context = 0
        for paragraph in raw_paragraphs:
            raw_text = paragraph.text
            text = re.sub(r'\[\d+\]', '', raw_text)
            if paragraph.name == 'p':
                nb_added_contexts, added_contexts = context_delimiter(text, 100000000)
                for id_context_added, added_context in enumerate(added_contexts):
                    context = {'id_context': id_context + id_context_added, 'text': added_context}
                    contexts.append(context)
                id_context += nb_added_contexts
            else:
                last_context = contexts.pop()
                nb_added_contexts, added_contexts = context_delimiter(last_context['text'] + text, 100000000)
                if nb_added_contexts == 1:
                    last_context['text'] += text
                    contexts.append(last_context)
                else:
                    id_context -= 1
                    for id_context_added, added_context in enumerate(added_contexts):
                        context = {'id_context': id_context + id_context_added, 'text': added_context}
                        contexts.append(context)
                    id_context += nb_added_contexts
        article['contexts'] = contexts
        articles.append(article)
        print("Article {}/{} traité !".format(article_id, nb_articles))
    return articles


def load_into_pickle_file(articles):
    with open('list_articles.pickle', 'wb') as pickle_file:
        pickle.dump(articles, pickle_file)


if __name__ == '__main__':
    wiki_article = wikipedia_crawler(base_url_wikipedia)
    load_into_pickle_file(wiki_article)




