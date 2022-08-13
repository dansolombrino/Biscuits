import os
from os import listdir
from os.path import isfile, join
from pprint import pprint
from PIL import Image


from tqdm import tqdm

subfolders = [ 
    f.path for f in os.scandir(
        "/home/dansolombrino/GitHub/biscuits/data/food-101_10_classes_in"
    ) if f.is_dir() 
]
to_resize = list()

for folder in subfolders:
    files_in_folder = [
        f.path for f in os.scandir(
            folder
        ) if f.is_file()
    ]

    to_resize.append(files_in_folder)


for rs in tqdm(to_resize):
# for rs in to_resize:
    # pprint(rs)
    for f in tqdm(rs):
    # for f in rs:
        f_replaced = f
        f_replaced = f_replaced.replace(
            "food-101_10_classes_in", "food-101_10_classes_resized"
        )

        path = '/'.join(f_replaced.split('/')[:-1])

        if not os.path.exists(path):
            os.makedirs(path)

        img = Image.open(f)
        new_width  = 64
        new_height = 64
        img = img.resize(
            (new_width, new_height), 
            Image.ANTIALIAS
        )
        img.save(f_replaced)
    

    