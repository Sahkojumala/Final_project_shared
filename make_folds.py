# make_folds.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='../data/HAM-10000/HAM10000_metadata.csv')
ap.add_argument('--out_dir', default='../data/HAM-10000')
ap.add_argument('--n_splits', type=int, default=5)
ap.add_argument('--label_col', default='dx')        # alkuperäinen diagnoosisarake
args = ap.parse_args()

df = pd.read_csv(args.csv)

# luokka-numero (0..6)
class2idx = {cls: idx for idx, cls in enumerate(sorted(df[args.label_col].unique()))}
df['label'] = df[args.label_col].map(class2idx)

# image_id – ilman .jpg-päätettä
if 'image_id' not in df.columns:
    df['image_id'] = df['image'].str.replace('.jpg', '', regex=False)

skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
    train_df = df.iloc[train_idx][['image_id', 'label']]
    val_df   = df.iloc[val_idx][['image_id', 'label']]
    train_df.to_csv(os.path.join(args.out_dir, f'train_fold{fold}.csv'), index=False)
    val_df.to_csv(os.path.join(args.out_dir, f'val_fold{fold}.csv'), index=False)

print(f"5 foldia tallennettu kansioon {args.out_dir}")