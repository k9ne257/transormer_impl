from pathlib import Path
import pandas as pd
import pyarrow
# import fastparquet

files = list(Path("../data/wmt_de_en").glob("*.parquet"))
print("Files:", len(files))

# print("File 1:", files[0])
df = pd.read_parquet(files[0])
print(df.head())
print(df.columns)


# parquet_files = list(Path("../data/wmt_de_en/").glob("*.parquet"))
# print("Found files:")
#
# for p in parquet_files:
#     print(p)
#
# pd.read_parquet(parquet_files[0])

# text = "\n".join(
#     " ".join(pd.read_parquet(p)["text"].astype(str))
#     for p in parquet_files
# )


# # here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# # create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


