# args.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],)
    # Käyttäjä voi valita, mitä ResNet-verkkoa käytetään mallin runkona.

    parser.add_argument('--csv_dir', type=str, default='../data/HAM-10000')     # kansio jossa foldit ovat
    # Hakemisto, josta CSV-data ladataan.

    parser.add_argument('--num_classes', type=int, default=7)
    # Luokkien määrä 7 

    parser.add_argument('--batch_size', type=int, default=32,
                        choices=[16, 32, 64])
    # Mini-batchin koko mallin harjoituksessa.

    parser.add_argument('--lr', type=float, default=1e-3,)
    # Oppimisnopeus (learning rate).

    parser.add_argument('--epochs', type=int, default=10,)
    # Epochien määrä
    
    parser.add_argument('--out_dir', type=str, default='ham_sessions' )
    # Hakemisto, johon mallin tulokset/tarkistuspisteet tallennetaan.

    
    return parser.parse_args()
