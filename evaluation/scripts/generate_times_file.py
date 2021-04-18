import pandas as pd
from eval_preproc import FramePreprocessor

filenames = pd.read_csv("C:/Users/julia/Downloads/data.csv")
fp = FramePreprocessor(None, None, None, None)
filenames.filename = filenames["#timestamp [ns]"].apply(fp.standardize_timestamp)
filenames.to_csv("C:/Users/julia/Downloads/times.txt", header=False, index=False, sep="\t")
