# trainer.py

import os
from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm           # edistymis palkki
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, fold) :
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)                                        # tallennuskansio sessions. luo automaattisesti jos ei ole

    # Cuda tai CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)  # pieni näkyvyys
    model.to(device)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    early_stop_patience = 8
    epochs_no_improve = 0

    best_bal_acc = float('-inf')                                                    # <-- paras tähän asti

    # historian keruu
    hist = {
        'train_loss': [], 'val_loss': [],
        'train_bal_acc': [], 'val_bal_acc': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_avg_prec': [], 'val_avg_prec': [],
    }

    for epoch in range(args.epochs):
        training_loss = 0.0
        # starting the training -> setting the model to training mode
        model.train()

        # kerää train-ennusteet metriikoita varten
        y_true_tr, y_pred_tr, y_prob_tr = [], [], []
        num_classes = None

        for batch in train_loader:                                                  # simppeli versio ilman edistymis palkkia
        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):   # edistymis palkki
            inputs = batch['img'].to(device, dtype=torch.float32)                   # varmistaa että samalla laitteella ja että float32
            targets = batch['label'].to(device)                                     # varmistaa että samalla laitteella

            # Resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # kerää train-metriikoihin
            probs = F.softmax(outputs, dim=1)        # (B, C)
            preds = torch.argmax(probs, dim=1)       # (B,)
            if num_classes is None:
                num_classes = probs.shape[1]
            y_true_tr.append(targets.detach().cpu().numpy())
            y_pred_tr.append(preds.detach().cpu().numpy())
            y_prob_tr.append(probs.detach().cpu().numpy())

            # backprop
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # laske train-metriikat epochin lopussa
        train_avg_loss = training_loss / len(train_loader)
        y_true_tr = np.concatenate(y_true_tr, axis=0)
        y_pred_tr = np.concatenate(y_pred_tr, axis=0)
        y_prob_tr = np.concatenate(y_prob_tr, axis=0)
        classes = list(range(num_classes)) if num_classes is not None else [0,1,2,3,4]

        train_bal_acc = balanced_accuracy_score(y_true_tr, y_pred_tr)
        Y_true_bin_tr = label_binarize(y_true_tr, classes=classes)
        try:
            train_roc_auc = roc_auc_score(y_true_tr, y_prob_tr, average='macro', multi_class='ovr')
            # train_roc_auc = roc_auc_score(Y_true_bin_tr, y_prob_tr, average='macro', multi_class='ovr')
        except Exception:
            train_roc_auc = np.nan
        try:
            train_avg_prec = average_precision_score(Y_true_bin_tr, y_prob_tr, average='macro')
        except Exception:
            train_avg_prec = np.nan

        print('Epoch-{}: {:.4f}'.format(epoch + 1, train_avg_loss))
        print('Train:      loss={:.4f} | bal_acc={:.4f} | roc_auc={} | avg_prec={}'.format(
            train_avg_loss, train_bal_acc,
            'nan' if np.isnan(train_roc_auc) else f'{train_roc_auc:.4f}',
            'nan' if np.isnan(train_avg_prec) else f'{train_avg_prec:.4f}'
        ))


        val_loss, val_bal_acc, val_roc_auc, val_avg_prec = validate_model(model, val_loader, criterion)
        print('Validation: loss={:.4f} | bal_acc={:.4f} | roc_auc={} | avg_prec={}'.format(
            val_loss, val_bal_acc,
            'nan' if np.isnan(val_roc_auc) else f'{val_roc_auc:.4f}',
            'nan' if np.isnan(val_avg_prec) else f'{val_avg_prec:.4f}'
        ))


        # -----------------------------
        # 7a: Scheduler + Early stop
        # -----------------------------
        scheduler.step(val_bal_acc)          # HUOM: mode='max' -> parannus = suurempi bal_acc

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint -> {ckpt_path} (bal_acc={val_bal_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        # -----------------------------



        # tallenna historiaan
        hist['train_loss'].append(train_avg_loss)
        hist['val_loss'].append(val_loss)
        hist['train_bal_acc'].append(train_bal_acc)
        hist['val_bal_acc'].append(val_bal_acc)
        hist['train_roc_auc'].append(train_roc_auc)
        hist['val_roc_auc'].append(val_roc_auc)
        hist['train_avg_prec'].append(train_avg_prec)
        hist['val_avg_prec'].append(val_avg_prec)

        # <-- Tallenna paras Balanced Accuracy
        # if val_bal_acc > best_bal_acc:
        #    best_bal_acc = val_bal_acc
        #    ckpt_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
        #    torch.save(model.state_dict(), ckpt_path)
        #    print(f"Saved best checkpoint -> {ckpt_path} (bal_acc={val_bal_acc:.4f})")

        # val_loss = validate_model(model, val_loader, criterion)
        # print('Validation loss: {}'.format(val_loss))

    # piirrä ja tallenna 4 kuvaa per fold
    _plot_curves(hist, args.out_dir, fold)

    # tallenna numerot CSV:ksi
    pd.DataFrame(hist).to_csv(os.path.join(args.out_dir, f"metrics_fold{fold}.csv"), index=False)



def validate_model(model, val_loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    val_loss = 0.0

    y_true = []
    y_pred = []
    y_prob = []     # todennäköisyydet (N x C)
    num_classes = None

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device, dtype=torch.float32)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            # häviö
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # metriikoita varten
            probs = F.softmax(outputs, dim=1)          # (B, C)
            preds = torch.argmax(probs, dim=1)         # (B,)

            if num_classes is None:
                num_classes = probs.shape[1]

            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())

    # yhdistä batchit
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)            # (N, C)
    classes = list(range(num_classes)) if num_classes is not None else [0,1,2,3,4]

    # Balanced Accuracy
    val_bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Binarisointi ROC-AUC / Average Precisionia varten
    Y_true_bin = label_binarize(y_true, classes=classes)   # (N, C)

    # ROC–AUC (macro, one-vs-rest)
    try:
        # val_roc_auc = roc_auc_score(Y_true_bin, y_prob, average='macro', multi_class='ovr')
        val_roc_auc = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
    except Exception:
        val_roc_auc = np.nan

    # Average Precision (macro)
    try:
        val_avg_prec = average_precision_score(Y_true_bin, y_prob, average='macro')
    except Exception:
        val_avg_prec = np.nan

    return (val_loss / len(val_loader), val_bal_acc, val_roc_auc, val_avg_prec)


# Apufunktio ploteille
def _plot_curves(hist, out_dir, fold):
    epochs = range(1, len(hist['train_loss']) + 1)

    def _save_plot(y_train, y_val, ylabel, fname):
        plt.figure()
        plt.plot(epochs, y_train, label='train')
        plt.plot(epochs, y_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fold{fold}_{fname}.png"))
        plt.close()

    _save_plot(hist['train_loss'],     hist['val_loss'],     'Loss',              'loss')
    _save_plot(hist['train_bal_acc'],  hist['val_bal_acc'],  'Balanced Accuracy', 'balacc')
    _save_plot(hist['train_roc_auc'],  hist['val_roc_auc'],  'ROC-AUC',           'rocauc')
    _save_plot(hist['train_avg_prec'], hist['val_avg_prec'], 'Average Precision', 'avgprec')




"""
def validate_model(model, val_loader, criterion):
    model.eval()
    device = next(model.parameters()).device                        # poimi mallin laite
    val_loss = 0.0
    with torch.no_grad():                                           # evalissa ei tarvita gradientteja
        for batch in val_loader:
            inputs = batch['img'].to(device, dtype=torch.float32)
            targets = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# alkuperäinen simppeli malli

def validate_model(model, val_loader, criterion):
    model. eval()
    val_loss = 0

    for batch in val_loader:
        inputs = batch['img']
        targets = batch['label']

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss. item()


    return val_loss / len(val_loader)
"""





