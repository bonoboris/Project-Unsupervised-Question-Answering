import bs4
import requests
import re

root_url = 'http://fr.wikipedia.org'
page_url = root_url + '/wiki/Jacques_Chirac'

if __name__ == '__main__':
    request = requests.get(page_url)
    soup = bs4.BeautifulSoup(request.content, features="html.parser")
    fp = open('data_wiki.txt', 'w')
    text = soup.select('div p')
    for paragraph in text:
        text_paragraph = re.sub(r"\[.+\]", "", paragraph.get_text())
        fp.write(text_paragraph)
    fp.close()
