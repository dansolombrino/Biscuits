from importlib.metadata import files
import os
from os import listdir
from os.path import isfile, join
from pprint import pprint
import random
import shutil

TRAIN_PERCENTAGE = 0.6
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 1 - TRAIN_PERCENTAGE - VALIDATION_PERCENTAGE

from tqdm import tqdm

subfolders = [ 
    f.path for f in os.scandir(
        "/home/dansolombrino/GitHub/biscuits/data/EuroSAT_X_food-101_in"
    ) if f.is_dir() 
]

folders = dict()
folder_indices = dict()

for folder in subfolders:
    files_in_folder = [
        f.path for f in os.scandir(
            folder
        ) if f.is_file()
    ]

    random.shuffle(files_in_folder)

    folders[folder] = files_in_folder

    files_indices = range(0, len(files_in_folder))

    train_indices = files_indices[
        0: int(len(files_indices) * TRAIN_PERCENTAGE)
    ]

    validation_indices = files_indices[
        int(len(files_indices) * TRAIN_PERCENTAGE) : 
        int (len(files_indices) * (TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE))
    ]
    
    test_indices = files_indices[
        int(
            len(files_indices) * (TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE)
        ) : 
    ]

    folder_indices[folder] = dict()
    folder_indices[folder]["train"] = train_indices
    folder_indices[folder]["val"] = validation_indices
    folder_indices[folder]["test"] = test_indices

    # print(train_indices[0], " --> ", train_indices[-1])
    # print(validation_indices[0], " --> ", validation_indices[-1])
    # print(test_indices[0], " --> ", test_indices[-1])

    # break


# pprint(folder_indices)
# pprint(folders[list(folders.keys())[0]])


# for k in folders.keys():
for k in tqdm(folders.keys()):
    
    # print(k)

    k_dict = dict()
    
    k_train = k.replace(
        "EuroSAT_X_food-101_in/", 
        "EuroSAT_X_food-101_splitted/train/"
    )
    if not os.path.exists(k_train):
        os.makedirs(k_train)

    k_validation = k.replace(
        "EuroSAT_X_food-101_in/", 
        "EuroSAT_X_food-101_splitted/val/"
    )
    if not os.path.exists(k_validation):
        os.makedirs(k_validation)
    
    k_test = k.replace(
        "EuroSAT_X_food-101_in/", 
        "EuroSAT_X_food-101_splitted/test/"
    )
    if not os.path.exists(k_test):
        os.makedirs(k_test)

    k_dict["original"] = k
    k_dict["train"] = k_train
    k_dict["val"] = k_validation
    k_dict["test"] = k_test

    # print(k_train)
    # print(k_validation)
    # print(k_test)
    # print("\n\n\n")

    # print(folder_indices[k].keys())

    for ind in list(folder_indices[k].keys()):
        
        src = folders[k][
            folder_indices[k][ind][0] : folder_indices[k][ind][-1]
        ]
        # pprint(src[0: 3])

        dst = k_dict[ind]
        if not os.path.exists(dst):
            os.makedirs(dst)
        # print(dst)

        for f in src:
            # print(f)
            
            f_name = f.split("/")[-1]
            # print(dst + "/" + f_name)
            
            # print("\n\n")

            shutil.copyfile(f, dst + "/" + f_name)