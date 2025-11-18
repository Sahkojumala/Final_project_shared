import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("../data/PAD-UFES-20/metadata.csv"))

# 1) luokkajakauma (tässä sarake on 'diagnostic')
print("Luokkajakauma (diagnostic):")
print(df["diagnostic"].value_counts(), "\n")

# 2) sukupuoli- ja ikäjakauma
print("Sukupuoli:")
print(df["gender"].value_counts(dropna=False), "\n")

print("Ikä – perusstatistiikka:")
print(df["age"].describe(), "\n")

# 3) esimerkkirivit luokittain
print("Esimerkkejä per luokka:")
print(df.groupby("diagnostic").head(1)[["img_id","diagnostic","age","gender"]])
