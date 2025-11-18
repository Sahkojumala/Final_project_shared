
# evaluate.py

import os, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             average_precision_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize
from dataset import HAMDataset
from torchvision import transforms
from models import MyModel
from args import get_args
import seaborn as sns
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def evaluate_fold(fold):
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # lataa testi-CSV (käytetään val-foldia testinä)
    test_df = pd.read_csv(os.path.join(args.csv_dir, f"val_fold{fold}.csv"))
    test_set = HAMDataset(test_df, root_dir=args.csv_dir, transform=val_tf)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = MyModel(args.backbone).to(device)
    ckpt = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            y_true.append(targets.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # metriikat
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
    avg_prec = average_precision_score(label_binarize(y_true, classes=range(args.num_classes)),
                                       y_prob, average='macro')
    print(f"Fold {fold} – Bal.Acc: {bal_acc:.4f}  ROC-AUC: {roc_auc:.4f}  AvgPrec: {avg_prec:.4f}")
    print(classification_report(y_true, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion matrix – fold {fold}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(args.out_dir, f'cm_fold{fold}.png'))
    plt.close()

if __name__ == '__main__':
    for f in range(1,6):
        evaluate_fold(f)