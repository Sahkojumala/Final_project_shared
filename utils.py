# utils.py


# ???

# alu puuttuua ja loppu

# ???

def save_checkpoint(cur_fold, cur_epoch, model, y_true, y_pred, val_metric,
                    best_val_metric=None, prev_model_path=None,
                    comparator='gt', save_dir='sessions'):
# init(autoreset=True) # linux

    os.makedirs(save_dir, exist_ok=True)
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(save_dir, f'fold_{cur_fold}_epoch_{cur_epoch+1}.pth')

    if best_val_metric is None or comparator(val_metric, best_val_metric):
        if prev_model_path and os.path.exists(prev_model_path):
            os.remove(prev_model_path)

        torch.save(model.state_dict(), cur_snapshot_name)
        print(' ==== > Snapshot saved to: ', 'green' + cur_snapshot_name, 'yellow')
        print(f'Validation metric improved to: {val_metric :. 4f}', 'cyan')

        session_info = {
            'fold_id': cur_fold,
            'epoch': cur_epoch,
            'best_val_metric': val_metric,
            'best_val_preds': y_pred,
            'best_val_target': y_true,
            'model_path': cur_snapshot_name
        }



        filename = os.path. join(save_dir, f'session_fold_{cur_fold}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(session_info, f)

        return val_metric, cur_snapshot_name
    else:
        return best_val_metric, prev_model_path
    


    # ???


    