# This file loads the urls into the txt folder

from bs4 import BeautifulSoup
import urllib3


def url_crawler():
    URL = 'https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Article_de_qualit%C3%A9'

    base_url = 'https://fr.wikipedia.org/w/index.php?title=Cat%C3%A9gorie:Article_de_qualit%C3%A9&from='

    letter_list = ['0']
    for ascii_code in range(65, 65 + 26):
        letter_list.append(chr(ascii_code))

    http = urllib3.PoolManager()

    principal_page = http.request('GET', URL).data.decode('utf-8')

    soup = BeautifulSoup(principal_page, 'html.parser')

    urls_list = []

    for letter in letter_list:
        html_doc = http.request('GET', base_url + letter).data.decode('utf-8')
        soup = BeautifulSoup(html_doc, 'html.parser')
        category_div = soup.find("div", class_="mw-category-group")
        title = category_div.find("h3").text[0]
        if title == letter:
            links = category_div.find_all("a")
            urls = list(map(lambda x: x['href'], links))
            urls_list.extend(urls)

    with open('urls_articles_qualites.txt', 'w') as file:
        for url in urls_list:
            file.write(url + '\n')

    return urls_list
# This is the URL of good quality articles

