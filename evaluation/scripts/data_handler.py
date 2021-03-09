import urllib.request
from progress_bar import download_progress as dp
import zipfile
import os
import shutil


class DataHandler:
    def __init__(self, filename, url, dest="tmp"):
        self.filename = filename
        self.url = url
        self.dest = dest
        
    def download(self):
        if not os.path.exists(self.dest):
            os.mkdir(self.dest)
        filepath, _ = urllib.request.urlretrieve(self.url, os.path.join(self.dest, self.filename + ".zip"), dp())
        
    def unzip(self):
        with zipfile.ZipFile(os.path.join(self.dest, self.filename + ".zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.dest, self.filename))
            
    def delete_zip(self):
        os.remove(os.path.join(self.dest, self.filename + ".zip"))

    def clean_download_dir(self):
        shutil.rmtree(self.dest)
        
        
if __name__ == "__main__":
    url = 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip'
    dh = DataHandler("test", url)
    dh.download()
    dh.unzip()
    dh.delete_zip()