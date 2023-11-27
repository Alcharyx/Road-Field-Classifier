from pathlib import Path
import shutil
import random as rd

def generate_dataset(data_path : Path,
                     data_folders : list,
                     data_split : list = [80,20]):
    """Split an organise dataset for training.
    Test part is done manually because no way to tell which class with default constitution

    Args:
        data_path (Path): Path of the dataset folder
        data_folders (list): list containing the name of the folder inside the data_path that contains train img, allows to identify class
        data_split (list): list containing the % split between val and train data (sum = 100)
    """
    if (data_split[0] + data_split[1] != 100):
        raise ValueError("Sum of split must be equal to 100")

    data_dir = []
    proper_dataset = Path("./dataset")
    if (proper_dataset).exists():
        shutil.rmtree(str(proper_dataset))
    proper_dataset.mkdir(parents = False, exist_ok = False)
    (proper_dataset/"train").mkdir(parents = False, exist_ok = False)
    (proper_dataset/"val").mkdir(parents = False, exist_ok = False)
    min_per_class = 10000
    for img_class in data_folders: #balance dataset (undersampling)
        if len(list((data_path / img_class).iterdir())) < min_per_class:
            min_per_class = len(list((data_path / img_class).iterdir()))
    amount_in_val = round(data_split[1] * min_per_class/100)
    
    for img_class in data_folders:
        img_counter = 1
        list_img = list((data_path / img_class).iterdir())
        for img_path in list_img:
            if img_counter <= min_per_class:
                if (img_counter <= amount_in_val):
                    shutil.copy(str(img_path),str(proper_dataset/"val"/(img_class+ "_" + str(img_counter)+".jpg")))
                else:
                    shutil.copy(str(img_path),str(proper_dataset/"train"/(img_class+ "_" + str(img_counter)+".jpg")))
                img_counter +=1


