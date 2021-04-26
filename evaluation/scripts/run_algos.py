import os
from sys import platform
import pandas as pd
from data_handler import DataHandler
from cli_command_gen import CommandGenerator
import shutil
from eval_preproc import FramePreprocessor
from position_evaluator import Evaluator
import subprocess
import time
from json_helper import JsonHelper
from image_helper import ImageHelper

# TODO: make script executable with params
PATH_TO_ORB_SLAM = "~/ORB_SLAM3"
PATH_TO_TMP_DIR = "../tmp"
PATH_TO_RESULT_DIR = "../results"
PATH_TO_CONFIG = "../config"
PATH_TO_DSM = "~/dsm"
PATH_TO_DSO_SLAM = "~/dso_with_saving_pcl"

if __name__ == "__main__":
    data_metadata = pd.read_csv(os.path.join(PATH_TO_CONFIG, "data_meta.csv"), sep=";")

    # itarate throu each dataset (=row in df) and run all algorithms
    for i in range(data_metadata.shape[0]):
        filename = data_metadata.iloc[i, 0]
        url = data_metadata.iloc[i, 1]
        dataset = data_metadata.iloc[i, 2]
        run = data_metadata.iloc[i, 3]
        eval_resolution = data_metadata.iloc[i, 4]

        ###############################
        # get the data
        ###############################
        if not run:
            print("skipping {}...".format(filename))
            continue

        for resolution in [1, 0.8, 0.6, 0.4]:
            if (resolution != 1) and (not eval_resolution):
                continue

            if resolution == 1 and not os.path.exists(PATH_TO_TMP_DIR + "/" + filename):
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
            if platform == "linux" or platform == "linux2":
                try:
                    ##################### ORB ###########################
                    res_orb_dir = os.path.join(PATH_TO_RESULT_DIR, filename, "ORB")
                    if resolution != 1:
                        res_orb_dir = os.path.join(PATH_TO_RESULT_DIR, filename + str(resolution), "ORB")
                    res_orb_data_dir = os.path.join(res_orb_dir, "data")

                    if not os.path.exists(res_orb_dir):
                        os.makedirs(res_orb_dir, exist_ok=True)
                    if not os.path.exists(res_orb_data_dir):
                        os.makedirs(res_orb_data_dir, exist_ok=True)

                    if resolution < 1:
                        # downsize resolution
                        print("downsizing resolution to " + str(resolution *100) + " percent")
                        image_helper = ImageHelper()
                        image_helper.resize_directory(os.path.join(PATH_TO_TMP_DIR,
                                                                   filename,
                                                                   "mav0",
                                                                   "cam0",
                                                                   "data"),
                                                      int(752 * resolution),
                                                      int(480 * resolution))

                    cg = CommandGenerator()
                    command = cg.orb(filename=filename,
                                     path_to_orb=PATH_TO_ORB_SLAM,
                                     path_to_data=PATH_TO_TMP_DIR,
                                     path_to_config=PATH_TO_CONFIG,
                                     dataset=dataset,
                                     resolution=resolution)
                    print("Running ORB slam on {}!".format(filename))
                    t1 = time.perf_counter()
                    process = subprocess.Popen(command, shell=True)
                    process.wait()
                    t2 = time.perf_counter()

                    # write the elapsed time to json file
                    elapsed = t2 - t1
                    jh = JsonHelper()
                    jh.add_json(os.path.join(res_orb_data_dir, "results.txt"), "processing_time", elapsed)

                    # try to copy the output in the right place
                    try:
                        shutil.move("KeyFrameTrajectory.txt", os.path.join(res_orb_data_dir, "estimated_data.txt"))
                        shutil.move("PointCloud.txt", os.path.join(res_orb_data_dir, "PointCloud.txt"))
                        point_cloud_gt_path = os.path.join(PATH_TO_TMP_DIR, filename, "mav0", "pointcloud0", "data.ply")
                        if os.path.exists(point_cloud_gt_path):
                            shutil.move(point_cloud_gt_path, os.path.join(res_orb_data_dir, "PointCloud_gt.ply"))

                    except:
                        raise ValueError("Something went wrong! Could not copy output files!")
                    preproc = FramePreprocessor(gt_filepath=os.path.join(PATH_TO_TMP_DIR, filename, "mav0",
                                                                         "state_groundtruth_estimate0", "data.csv"),
                                                est_filepath=os.path.join(res_orb_data_dir, "estimated_data.txt"),
                                                dataset_type=dataset,
                                                outdir_data=res_orb_data_dir)
                    print("reading in the result dataframes...")
                    preproc.create_est_pos_df()
                    preproc.create_gt_pos_df()
                    print("aligning the timestamps...")
                    preproc.align_timestamps()
                    preproc.get_gt_pos_df().to_csv(os.path.join(res_orb_data_dir, "position_gt.csv"))
                    print("transforming the coordinate system...")
                    preproc.transform_coordinate_system(outdir=res_orb_data_dir)


                    ##################### DSO ###########################
                    res_dso_dir = os.path.join(PATH_TO_RESULT_DIR, filename, "DSO")
                    if resolution == 1:
                        res_dso_data_dir = os.path.join(res_dso_dir, "data")

                        if not os.path.exists(res_dso_dir):
                            os.makedirs(res_dso_dir, exist_ok=True)
                        if not os.path.exists(res_dso_data_dir):
                            os.makedirs(res_dso_data_dir, exist_ok=True)

                        cg = CommandGenerator()
                        command = cg.dso(filename=filename,
                                         path_to_dso=PATH_TO_DSO_SLAM,
                                         path_to_data=PATH_TO_TMP_DIR,
                                         path_to_config=PATH_TO_CONFIG,
                                         dataset=dataset,
                                         resolution=resolution)
                        print("Running DSO slam on {}!".format(filename))
                        t1 = time.perf_counter()
                        process = subprocess.Popen(command, shell=True)
                        process.wait()
                        t2 = time.perf_counter()

                        # write the elapsed time to json file
                        elapsed = t2 - t1
                        jh = JsonHelper()
                        jh.add_json(os.path.join(res_dso_data_dir, "results.txt"), "processing_time", elapsed)

                        # try to copy the output in the right place
                        try:
                            shutil.move("result.txt", os.path.join(res_dso_data_dir, "estimated_data.txt"))
                            shutil.move("pcl_data.pcd", os.path.join(res_dso_data_dir, "PointCloud.txt"))

                        except:
                            raise ValueError("Something went wrong! Could not copy output files!")
                        preproc = FramePreprocessor(gt_filepath=os.path.join(PATH_TO_TMP_DIR, filename, "mav0",
                                                                             "state_groundtruth_estimate0", "data.csv"),
                                                    est_filepath=os.path.join(res_dso_data_dir, "estimated_data.txt"),
                                                    dataset_type=dataset,
                                                    outdir_data=res_dso_data_dir)
                        print("reading in the result dataframes...")
                        preproc.create_est_pos_df()
                        preproc.create_gt_pos_df()
                        print("aligning the timestamps...")
                        preproc.align_timestamps()
                        preproc.get_gt_pos_df().to_csv(os.path.join(res_dso_data_dir, "position_gt.csv"))
                        print("transforming the coordinate system...")
                        preproc.transform_coordinate_system(outdir=res_dso_data_dir)


                   ##################### DSM ###########################
                    if resolution == 1:
                        res_dsm_dir = os.path.join(PATH_TO_RESULT_DIR, filename, "DSM")
                        res_dsm_data_dir = os.path.join(res_dsm_dir, "data")

                        if not os.path.exists(res_dsm_dir):
                            os.makedirs(res_dsm_dir, exist_ok=True)
                        if not os.path.exists(res_dsm_data_dir):
                            os.makedirs(res_dsm_data_dir, exist_ok=True)

                        cg = CommandGenerator()
                        command = cg.dsm(filename=filename,
                                         path_to_orb=PATH_TO_ORB_SLAM,
                                         path_to_data=PATH_TO_TMP_DIR,
                                         path_to_config=PATH_TO_CONFIG,
                                         dataset=dataset,
                                         resolution=resolution,
                                         path_to_dsm=PATH_TO_DSM)
                        print("Running DSM slam on {}!".format(filename))
                        t1 = time.perf_counter()
                        process = subprocess.Popen(command, shell=True)
                        process.wait()
                        t2 = time.perf_counter()

                        # write the elapsed time to json file
                        elapsed = t2 - t1
                        jh = JsonHelper()
                        jh.add_json(os.path.join(res_dsm_data_dir, "results.txt"), "processing_time", elapsed)

                        # try to copy the output in the right place
                        try:
                            shutil.move("result.txt",
                                        os.path.join(res_dsm_data_dir, "estimated_data.txt"))
                            shutil.move("PointCloud.ply", os.path.join(res_dsm_data_dir, "PointCloud.ply"))

                        except:
                            raise ValueError("Something went wrong! Could not copy output file!")
                        preproc = FramePreprocessor(gt_filepath=os.path.join(PATH_TO_TMP_DIR, filename, "mav0",
                                                                             "state_groundtruth_estimate0", "data.csv"),
                                                    est_filepath=os.path.join(res_dsm_data_dir, "estimated_data.txt"),
                                                    dataset_type=dataset,
                                                    outdir_data=res_dsm_data_dir)
                        print("reading in the result dataframes...")
                        preproc.create_est_pos_df()
                        preproc.create_gt_pos_df()
                        print("aligning the timestamps...")
                        preproc.align_timestamps()
                        preproc.get_gt_pos_df().to_csv(os.path.join(res_dsm_data_dir, "position_gt.csv"))
                        print("transforming the coordinate system...")
                        preproc.transform_coordinate_system(outdir=res_dsm_data_dir)

                    else:
                        print("Can't run DSM on lower resolution, so skipping...")

                    if (not eval_resolution) or (resolution == 0.4):
                        print("cleaning up the download directory...")

                except:
                    if os.path.exists(PATH_TO_TMP_DIR):
                        pass

                    raise ValueError("Could not run Slam")

            else:
                print("Slam algorithms can only be run on linux; skipping...")
