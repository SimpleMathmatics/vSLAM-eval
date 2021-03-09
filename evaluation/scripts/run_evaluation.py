import os
from sys import platform
import pandas as pd
from data_handler import DataHandler
from cli_command_gen import CommandGenerator
import shutil
from eval_preproc import FramePreprocessor
from position_evaluator import Evaluator


PATH_TO_METADATA = "../config/data_meta.xlsx"
PATH_TO_ORB_SLAM = "~/ORB-slam3"
PATH_TO_TMP_DIR = "../tmp"
PATH_TO_RESULT_DIR = "../results"

if __name__ == "__main__":
    data_metadata = pd.read_excel(PATH_TO_METADATA)

    # itarate throu each dataset (=row in df) and run all algorithms
    for i in range(data_metadata.shape[0]):
        filename = data_metadata.iloc[0, 0]
        url = data_metadata.iloc[0, 1]
        dataset = data_metadata.iloc[0, 2]

        ###############################
        # get the data
        ###############################
        dh = DataHandler(filename=filename, url=url, dest=PATH_TO_TMP_DIR)
        print("downloading {}...".format(filename))
        dh.download()
        print("unzipping {}...".format(filename))
        dh.unzip()
        print("deleting zip file...")
        dh.delete_zip()

        ################################
        # run algorithms
        ################################
        cg = CommandGenerator()
        if platform == "linux" or platform == "linux2":
            ##################### ORB ###########################
            command = cg.orb(filename=filename,
                             path_to_orb=PATH_TO_ORB_SLAM,
                             path_to_data=PATH_TO_TMP_DIR,
                             dataset=dataset)
            print("Running ORB slam on {}!".format(filename))
            os.system(command)
            res_orb_dir = os.path.join(PATH_TO_RESULT_DIR, filename, "ORB")
            res_orb_img_dir = os.path.join(res_orb_dir, "img")
            res_orb_raw_dir = os.path.join(res_orb_dir, "raw_out")
            if not os.path.exists(res_orb_dir):
                os.mkdir(res_orb_dir)
            if not os.path.exists(res_orb_img_dir):
                os.mkdir(res_orb_img_dir)
            if not os.path.exists(res_orb_raw_dir):
                os.mkdir(res_orb_raw_dir)
            # try to copy the output in the right place
            try:
                shutil.move("f_dataset-{}_mono.txt".format(filename), os.path.join(res_orb_raw_dir, "estimated_data.txt"))
            except:
                raise ValueError("Something went wrong! Could not copy output file!")
            preproc = FramePreprocessor(gt_filepath=os.path.join(PATH_TO_TMP_DIR, filename, "mav0", "cam0",
                                                                 "state_groundtruth_estimate0", "data.csv"),
                                        est_filepath=os.path.join(res_orb_raw_dir, "estimated_data.txt"),
                                        dataset_type=dataset)
            print("reading in the result dataframes...")
            preproc.create_est_pos_df()
            preproc.create_gt_pos_df()
            print("aligning the timestamps")
            preproc.align_timestamps()
            print("transforming the coordinate system...")
            preproc.transform_coordinate_system()
            print("evaluating...")
            position_eval = Evaluator(gt_df=preproc.get_gt_pos_df(),
                                      est_df=preproc.get_est_pos_df())
            position_eval.create_pos_dif_plots(outdir=res_orb_img_dir)
            print("cleaning up the download directory...")
            dh.clean_download_dir()

        else:
            print("Slam algorithms can only be run on linux; skipping...")

