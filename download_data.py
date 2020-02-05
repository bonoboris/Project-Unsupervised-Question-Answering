import sys
import time
import urllib
import os
import shutil
import subprocess
import urllib

import nltk

from uqa.models import MODELS_PATH


nltk.download("punkt")
nltk.download("stopwords")
print()
subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
print()

file_urls = {
    "frwiki.gensim": "https://zenodo.org/record/162792/files/frwiki.gensim?download=1",
    "frwiki.gensim.syn0.npy": "https://zenodo.org/record/162792/files/frwiki.gensim.syn0.npy?download=1",
    "frwiki.gensim.syn1neg.npy": "https://zenodo.org/record/162792/files/frwiki.gensim.syn1neg.npy?download=1",
}


class Downloader(object):
    """Download a distant ressource with urllib.urlretrieve"""
    DOWNLOAD_DIR = MODELS_PATH

    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
        self.filepath = os.path.realpath(os.path.join(self.DOWNLOAD_DIR, filename))

    def download_if_doesnt_exist(self):
        """Check if path_file exists, if not try to fetch the content from url and create the file with the content."""
        print(f'Downloading "{self.filepath}"...')
        if os.path.isfile(self.filepath):
            print(f"...skipping file \"{self.filepath}\" : already on disk.")
        else:
            self.download()

    def download(self):
        try:
            urllib.request.urlretrieve(self.url, self.filepath, Downloader.reporthook)
            print("...Done !")
        except urllib.error.URLError as e:
            if hasattr(e, 'reason'):
                sys.stderr.write(f'[ERROR]: Cannot reach the server: {e.reason}')
            elif hasattr(e, 'code'):
                sys.stderr.write(f'[ERROR]: Server code : {e.code}.')

    def __del__(self):
        urllib.request.urlcleanup()

    @staticmethod
    def reporthook(count, block_size, total_size):
        global start_time
        global last_progress_size
        global last_time
        global last_msg_size
        if count == 0:
            last_msg_size = 0
            last_progress_size = 0
            last_time = 0
            start_time = time.time()
            return

        current_time = time.time()
        delta_time = current_time - last_time
        if (delta_time < 1):
            return
        elapsed = Downloader.seconds_to_pretty(current_time - start_time)

        progress_size = int(count * block_size)
        progress_size_mb = float(progress_size) / (1024. * 1024.)
        total_size_mb = float(total_size) / (1024. * 1024.)
        progress = float(progress_size) / float(total_size)
        delta_size = progress_size - last_progress_size

        speed = float(delta_size) / float(delta_time)
        speed_kb = speed / 1024.
        remaining = Downloader.seconds_to_pretty((total_size - progress_size) / speed)
        msg = "\r...{:2.1%} - {:.1f}/{:.1f} MB, ({:.1f} KB/s) [{} elapsed/ {} remaining]".format(
            progress, progress_size_mb, total_size_mb, speed_kb, elapsed, remaining)
        sys.stdout.write(f"\r{' ' * last_msg_size}")
        sys.stdout.write(msg)
        sys.stdout.flush()
        last_msg_size = len(msg)
        last_progress_size = progress_size
        last_time = current_time

    @staticmethod
    def seconds_to_pretty(val):
        val = int(val)
        tot_min = int(val / 60)
        hours = int(tot_min / 60)
        mins = tot_min % 60
        secs = val % 60
        ret = f"{secs}s"
        if mins > 0 or hours > 0:
            ret = f"{mins}m " + ret
        if hours > 0:
            ret = f"{hours}h " + ret
        return ret


if __name__ == "__main__":

    for filename, url in file_urls.items():
        Downloader(url, filename).download_if_doesnt_exist()
        print("")
