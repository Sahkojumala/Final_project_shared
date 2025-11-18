from args import get_args
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from utils import save_checkpoint


def train_model(model, train_loader, val_loader, cur_fold):
    args = get_args()

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = None
    best_model_path = None

    for epoch in range(args.epochs):
        training_loss = 0
        # starting the training -> setting the model to training mode
        model.train()

        for batch in train_loader:
            inputs = batch['img']
            targets = batch['label']

            # Resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print('Epoch-{}: {}'.format((epoch + 1), training_loss / len(train_loader)))

        ba, y_true, y_pred = validate_model(model, val_loader, criterion)

        best_ba, best_model_path = save_checkpoint(cur_fold,
                        epoch,
                        model,
                        y_true,
                        y_pred,
                        ba,
                        best_val_metric = ba,
                        prev_model_path = best_model_path,
                        comparator='gt',
                        save_dir=args.out_dir)


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    for batch in val_loader:
        inputs = batch['img']
        targets = batch['label']

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()

        predictions = F.softmax(outputs, dim=1)
        pred_targets = predictions.max(dim=1)[1]

        all_preds.append(pred_targets.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ba = balanced_accuracy_score(all_preds, all_targets)

    return ba, all_targets, all_preds


