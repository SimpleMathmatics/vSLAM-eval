from PIL import Image
from os import listdir
from os.path import isfile, join
import os


class ImageHelper:
    @staticmethod
    def resize_image(path_to_image, size_x, size_y):
        image = Image.open(path_to_image)
        new_image = image.resize((size_x, size_y))
        new_image.save(path_to_image)

    def resize_directory(self, path_to_dir, size_x, size_y):
        files = [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]
        for f in files:
            f_path = os.path.join(path_to_dir, f)
            self.__class__.resize_image(f_path, size_x, size_y)



