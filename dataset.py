# dataset.py
import os, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset

class HAMDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.transform = transform

        # rakenna nopea lookup-dict: image_id → täysi polku
        self.id2path = {}
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                if f.lower().endswith('.jpg'):
                    fid = f.replace('.jpg', '')
                    self.id2path[fid] = os.path.join(dirpath, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.loc[idx, 'image_id'])
        label = int(self.df.loc[idx, 'label'])

        if image_id not in self.id2path:
            raise FileNotFoundError(f"Kuvaa {image_id}.jpg ei löydy alihakemistoista")

        img_path = self.id2path[image_id]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'label': label}