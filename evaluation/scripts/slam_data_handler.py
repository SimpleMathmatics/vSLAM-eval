import urllib.request
from progress_bar import download_progress as dp
import zipfile
import os


class data_handler:
    def __init__(self, filename, url, dest="tmp/"):
        self.filename = filename
        self.url = url
        self.dest = dest
        
    def download(self):
        filepath, _ = urllib.request.urlretrieve(self.url, self.dest + self.filename + ".zip", dp())
        
    def unzip(self):
        with zipfile.ZipFile(self.dest + self.filename + ".zip", 'r') as zip_ref:
            zip_ref.extractall(self.dest + self.filename)
            
    def delete_zip(self):
        os.remove(self.dest + self.filename + ".zip")
        
        
if __name__ == "__main__":
    url = 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip'
    dh = data_handler("test", url)
    #dh.download()
    dh.unzip()
    dh.delete_zip()
    