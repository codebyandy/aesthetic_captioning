import os
import re
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

import pandas as pd # Added by Andy (06/04/2023)
import pdb # Aded by Andy (06/04/2023)


def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def prepare_json(image_root, split):
    imgs = []
    for i, c in enumerate(json.load(open(image_root+"/processed_%s.json" % split, 'r'))):
        captions = [remove_URL(k) for k in c['first_level_comments_values']]
        imgs.append({'image_id': str(i), 'image': c['im_paths'], 'caption': captions})
    return imgs


def prepare_ann():
    """
    Returns list of {image_id, image, caption} as expected by self.annotation in class reddit_train()

    NOTE: Added by Andy (06/04/2023)
    - Exisitng method prepare_json() needs missing json, but I'm not sure how to get it

    returns
    - imgs: list of dicts, {image_id, image, caption}
    """
    df = pd.read_csv('../../../data/RPCD/dataset_andy.csv')
    
    img_captions_dict = {}
    
    for i, row in df.iterrows():
        img_path = row['img_paths']
        caption = row["comment_body"]

        if not isinstance(img_path, str):
            continue

        if img_path in img_captions_dict:
            img_captions_dict[img_path].append(caption)
            continue
        
        img_captions_dict[img_path] = [caption]

    imgs = []
    for i, (img, captions) in enumerate(img_captions_dict.items()):
        imgs.append({'image_id': str(i), 'image': img, 'caption': captions})

    return imgs


class reddit_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        # self.annotation = prepare_json(image_root, 'train') # Replace with new function in following line
        self.annotation = prepare_ann()
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):    

        ann = self.annotation[index]

        # image_path = os.path.join(self.image_root, ann['image'])  
        image_path = ann['image'] # added by andy      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(ann['caption'][np.random.randint(len(ann['caption']))], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 


class reddit_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''

        # self.annotation = prepare_json(image_root, split)
        self.annotation = prepare_ann()
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        # image_path = os.path.join(self.image_root, ann['image'])    
        image_path = ann['image'] # added by andy       
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image_id']
        
        return image, int(img_id)
