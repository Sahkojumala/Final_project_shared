

# ohjeita mäyttöön ja säätöön

# miten ajetaan komentoriviltä
# python3 main.py --backbone resnet18 --batch_size 16 --epochs 5 --csv_dir ../CSVs/folds

# kevyt ajo
cd "Data_Collection_Preparation/Knee Classification"
python3 main.py --backbone resnet18 --batch_size 16 --epochs 5 --csv_dir ../CSVs/folds


# The run in which the results were made
python3 main.py --backbone resnet18 --batch_size 32 --epochs 10

