import nltk

from os import path
import shutil
import subprocess
import urllib.request

nltk.download("punkt")
nltk.download("stopwords")
subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])

file_urls = {
    "uqa/models/frwiki.gensim": "https://zenodo.org/record/162792/files/frwiki.gensim?download=1",
    "uqa/models/frwiki.gensim.syn0.npy": "https://zenodo.org/record/162792/files/frwiki.gensim.syn0.npy?download=1",
    "uqa/models/frwiki.gensim.syn1neg.npy": "https://zenodo.org/record/162792/files/frwiki.gensim.syn1neg.npy?download=1",
}


def download_if_doesnt_exist(file_path, url):
    """Check if path_file exists, if not try to fetch the content from url and create the file with the content."""
    file_path = path.relpath(file_path)
    if not path.isfile(file_path):
        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    else:
        print(f"{file_path} already on disk... not downloading.")


for file_path, url in file_urls.items():
    download_if_doesnt_exist(file_path, url)
