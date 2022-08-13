import os
from os import listdir
from os.path import isfile, join
from pprint import pprint
import random
import shutil

from tqdm import tqdm

subfolders = [ 
    f.path for f in os.scandir(
        "/home/dansolombrino/GitHub/biscuits/data/EuroSAT_downsampled_in"
    ) if f.is_dir() 
]
random_selections = list()

for folder in subfolders:
    files_in_folder = [
        f.path for f in os.scandir(
            folder
        ) if f.is_file()
    ]

    random_selections.append(random.sample(files_in_folder, 120))


for rs in tqdm(random_selections):
    # pprint(rs)
    for f in tqdm(rs):
        f_replaced = f
        f_replaced = f_replaced.replace(
            "EuroSAT_downsampled_in", "EuroSAT_1200"
        )

        path = '/'.join(f_replaced.split('/')[:-1])

        if not os.path.exists(path):
            os.makedirs(path)

        shutil.copyfile(f, f_replaced)

    

    