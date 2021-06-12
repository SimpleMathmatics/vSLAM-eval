import os
import pandas as pd
import numpy as np
from glob import glob
import plotly.graph_objects as go
from position_evaluator import Evaluator
from point_cloud import PointCloud
import json


RESULT_DIR = "C:/Users/julia/Downloads/results"

# pq_1 = PointCloud()
# pq_1.from_file(os.path.join(RESULT_DIR, "V101", "ORB", "data_0", "PointCloud_gt.ply"))

# pq_2 = PointCloud()
# pq_2.from_file(os.path.join(RESULT_DIR, "V201", "ORB", "data_0", "PointCloud_gt.ply"))

def get_all_sub_dirs(resultdir):
    return(glob(resultdir + "/*/*/*/"))


def evaluate_res_dir(resultdir):
    all_sub_dirs = get_all_sub_dirs(resultdir)
    for dir_oi in all_sub_dirs:
        target_dir = os.path.join(dir_oi, "eval")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        print(dir_oi)
        evaluate_single_dir(dir_oi=dir_oi,
                            target_dir=target_dir)


def evaluate_single_dir(dir_oi, target_dir):
    print("test")
    # check if all necessary files are available
    files = os.listdir(dir_oi)
    if "estimated_data.txt" not in files:
        ValueError("Not all files present!")

    if "position_gt.csv" not in files:
        ValueError("Not all files present!")

    if "est_df_transformed.csv" not in files:
        ValueError("Not all files present!")

    if "rotation_matrix.npy" not in files:
        ValueError("Not all files present!")

    if "scale.npy" not in files:
        ValueError("Not all files present!")

    if "trans_vec.npy" not in files:
        ValueError("Not all files present!")

    if ("PointCloud.ply" not in files) and ("PointCloud.txt" not in files):
        ValueError("Not all files present!")

    #####
    # metadata
    #####

    # algo
    if dir_oi.__contains__("ORB"):
        algo = "ORB"
    elif dir_oi.__contains__("DSM"):
        algo = "DSM"
    elif dir_oi.__contains__("DSO"):
        algo = "DSO"
    else:
        ValueError("Path does not include algorithm name!")

    # sequence
    sequence_names = ["MH01", "MH02", "MH03", "MH04", "MH05",
                      "V101", "V102", "V102", "V103", "V201", "V202", "V203"]

    curr_oi = None
    for seq_oi in sequence_names:
        if dir_oi.__contains__(seq_oi):
            curr_seq = seq_oi
    if not curr_seq:
        ValueError("Path does not include sequence name!")

    # run
    curr_run = None
    runs = ["data_0", "data_1", "data_2"]
    for run_oi in runs:
        if dir_oi.__contains__(run_oi):
            curr_run = run_oi
    if not curr_seq:
        ValueError("Path does not include run name!")


    #####
    # trajectory analysis
    #####
    ev = Evaluator()
    est_pos = pd.read_csv(os.path.join(dir_oi, "est_df_transformed.csv"))
    gt_pos = pd.read_csv(os.path.join(dir_oi, "position_gt.csv"))
    dist_df = ev.calculate_diff(est_df=est_pos, gt_df=gt_pos)
    mean_pos_dif = np.mean(dist_df["dist"])
    print(mean_pos_dif)
    fig = go.Figure([go.Scatter(x=dist_df['timestamp'], y=dist_df['dist'])])
    fig.write_image(os.path.join(target_dir, "tray_dif.png"), engine="kaleido")

    # x and y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est_pos["p_x"], y=est_pos["p_y"],
                             mode='lines',
                             name='algo', line_width=0.5))
    fig.add_trace(go.Scatter(x=gt_pos["p_x"], y=gt_pos["p_y"],
                             mode='lines',
                             name='groundtruth', line_width=0.5))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
    )
    fig.write_image(os.path.join(target_dir, "tray_x_y.png"), engine="kaleido")


    # x and z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est_pos["p_x"], y=est_pos["p_z"],
                             mode='lines',
                             name='algo', line_width=0.5))
    fig.add_trace(go.Scatter(x=gt_pos["p_x"], y=gt_pos["p_z"],
                             mode='lines',
                             name='groundtruth', line_width=0.5))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="z",
    )
    fig.write_image(os.path.join(target_dir, "tray_x_z.png"), engine="kaleido")


    # y z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est_pos["p_y"], y=est_pos["p_z"],
                             mode='lines',
                             name='algo', line_width=0.5))
    fig.add_trace(go.Scatter(x=gt_pos["p_y"], y=gt_pos["p_z"],
                             mode='lines',
                             name='groundtruth', line_width=0.5))

    fig.update_layout(
        xaxis_title="y",
        yaxis_title="z",
    )
    fig.write_image(os.path.join(target_dir, "tray_y_z.png"), engine="kaleido")

    # Pointcloud
    pq_eval = PointCloud()
    if "PointCloud.txt" in files:
        pq_eval.from_file(os.path.join(dir_oi, "PointCloud.txt"))
    else:
        pq_eval.from_file(os.path.join(dir_oi, "PointCloud.ply"))
    rot = np.load(os.path.join(dir_oi, "rotation_matrix.npy"))
    trans = np.load(os.path.join(dir_oi, "trans_vec.npy"))
    scale = np.load(os.path.join(dir_oi, "scale.npy"))
    pq_eval.transform_points(trans=trans,
                             rot_mat=rot,
                             scale=scale)
    pq_errors = None
    if dir_oi.__contains__("V1"):
        pq_errors = pq_eval.compare_to_gt(gt_point_cloud=pq_1).tolist()
    elif dir_oi.__contains__("V2"):
        pq_errors = pq_eval.compare_to_gt(gt_point_cloud=pq_2).tolist()

    # computational time
    seq_length = {
        "MH01": 182,
        "MH02": 150,
        "MH03": 132,
        "MH04": 99,
        "MH05": 111,
        "V101": 144,
        "V102": 83.5,
        "V103": 105,
        "V201": 112,
        "V202": 115,
        "V203": 115
    }
    with open(os.path.join(dir_oi,'results.txt')) as json_file:
        results = json.load(json_file)
        proc_time = results["processing_time"]


    # write all results in json file
    result_dict = {
        "algorithm": algo,
        "sequence": curr_seq,
        "run": curr_run,
        "traj_timestamp": dist_df['timestamp'].tolist(),
        "traj_dist": dist_df["dist"].tolist(),
        "traj_x": est_pos["p_x"].tolist(),
        "traj_y": est_pos["p_y"].tolist(),
        "traj_z": est_pos["p_z"].tolist(),
        "traj_gt_x": gt_pos["p_x"].tolist(),
        "traj_gt_y": gt_pos["p_y"].tolist(),
        "traj_gt_z": gt_pos["p_z"].tolist(),
        "pq_errors": pq_errors,
        "n_points": pq_eval.n,
        "proc_time": proc_time,
        "seq_len": seq_length[curr_seq],
        "fps": seq_length[curr_seq] * 20/proc_time,
        "first_track": gt_pos["timestamp"].tolist()[0]
    }

    with open(os.path.join(dir_oi, "eval", "result.txt"), 'w') as outfile:
        json.dump(result_dict, outfile)


# evaluate_res_dir(RESULT_DIR)

target_dir = "C:/Users/julia/Downloads/results/MH01/ORB/data_1/eval"

print("pretest")

if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
evaluate_single_dir(dir_oi="C:/Users/julia/Downloads/results/MH01/ORB/data_1", target_dir=target_dir)