# main.py

from args import get_args
import os
import pandas as pd
# from dataset import Knee_Xray_datase
from dataset import HAMDataset
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model
from torchvision import transforms



def main():
    # 1. We need some arguments
    args = get_args ()

    # 2. Iterate among the folds
    for fold in range(1,6):
        print(f"Training fold: {fold}")

        train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold{fold}.csv"))    # Tiedostomuoto jota haetaan kansiosta
        val_set   = pd.read_csv(os.path.join(args.csv_dir, f"val_fold{fold}.csv"))
        

        


        # 
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 3. Preparing dataset
        # train_dataset = Knee_Xray_dataset(train_set) # vanhat datasetit
        # val_dataset = Knee_Xray_dataset(val_set)
        train_dataset = HAMDataset(train_set, root_dir=args.csv_dir, transform=train_tf)
        val_dataset   = HAMDataset(val_set,   root_dir=args.csv_dir, transform=val_tf)
        

        # 4. create dataloader
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        """
        # useampi datalanka nopeuttaa CPU-esikäsittelyä 
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        """
        
        # 5. initializing the model
        model = MyModel(args.backbone)

        # 6. train the model
        train_model(model, train_loader, val_loader, fold)


if __name__ == '__main__':
    main()
