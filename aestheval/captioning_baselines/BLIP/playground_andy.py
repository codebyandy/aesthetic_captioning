
"""
playground_andy.py
-
For testing modified BLIP methods
"""

from data.reddit_dataset import prepare_ann

from PIL import Image
import pandas as pd
import os

import pdb



IMG_DIR = '../../../data/RPCD/photocritique/images'

def add_img_path_dataset(): 
    """
    Take raw_reddit_photocritique_comments.csv and add column
    image path
    """
    
    def find_sub_dir(sub_id):
        for sub_dir in os.listdir(IMG_DIR):
            img_fns = os.listdir(os.path.join(IMG_DIR, sub_dir))
            # img_names = [img_fn[:img_fn.index('-')] for img_fn in img_fns]   
            for img_fn in img_fns:
                if sub_id in img_fn:
                    return sub_dir, img_fn
        return None, None

    df = pd.read_csv('../../../data/RPCD/raw_reddit_photocritique_comments.csv')

    img_paths = []
    id_path_dict = {}

    ext = set({})

    for i, row in df.iterrows():
        sub_id = row['submission_id']
        if sub_id in id_path_dict:
            img_paths.append(id_path_dict[sub_id])
            continue       
        
        sub_dir, img_fn = find_sub_dir(sub_id)
        
        if sub_dir is None:
            img_paths.append("NA")    
            continue
        if '.png' not in img_fn and '.jpeg' not in img_fn:
            img_paths.append("NA")    
            continue

        path = os.path.join(IMG_DIR, sub_dir)
        path = os.path.join(path, img_fn)
        id_path_dict[sub_id] = path

        img_paths.append(path)

    df['img_paths'] = img_paths
    df.to_csv('../../../data/RPCD/dataset_andy.csv', index=False)


def main():
    add_img_path_dataset()
    # prepare_ann()

if __name__ == "__main__":
    main()