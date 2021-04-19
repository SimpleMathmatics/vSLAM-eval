import os
from eval_preproc import FramePreprocessor

path_to_data = os.path.join("C:/Users/julia/Downloads/result.txt")
fp = FramePreprocessor(gt_filepath=None,
                       est_filepath=path_to_data,
                       dataset_type="euroc",
                       outdir_data=None)
fp.create_est_pos_df()
print(fp.get_est_pos_df())