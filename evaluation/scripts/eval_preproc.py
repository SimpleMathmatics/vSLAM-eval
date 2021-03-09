import pandas as pd
import numpy as np
from align import align_umeyama


class FramePreprocessor:
    def __init__(self, gt_filepath, est_filepath, dataset_type):
        self.dataset_type = dataset_type
        self.gt_filepath = gt_filepath
        self.est_filepath = est_filepath
        self.gt_pos_df = None
        self.est_pos_df = None

    def create_gt_pos_df(self):
        if self.dataset_type == "euroc":
            gt_df = pd.read_csv(self.gt_filepath, sep=",")
            
            # for now only use positional dataset_type
            gt_pos_df = pd.DataFrame({"timestamp": gt_df["#timestamp"],
                                      "p_x": gt_df[" p_RS_R_x [m]"],
                                      "p_y": gt_df[" p_RS_R_y [m]"],
                                      "p_z": gt_df[" p_RS_R_z [m]"]})
                                      
            self.gt_pos_df = gt_pos_df
            
    def create_est_pos_df(self):
        if self.dataset_type == "euroc":
            df_calc = pd.read_csv(self.est_filepath,
                                  sep=" ",
                                  header=None,
                                  names=["timestamp", "p_x", "p_y", "p_z", "1", "2", "3", "4"])
            self.est_pos_df = df_calc[["timestamp", "p_x", "p_y", "p_z"]]

    def align_timestamps(self):
        if all([isinstance(self.est_pos_df, pd.DataFrame), isinstance(self.gt_pos_df, pd.DataFrame)]):
            gt_ts = self.gt_pos_df["timestamp"].to_numpy()
            est_ts = self.est_pos_df["timestamp"].to_numpy()
            if not (np.all(np.diff(est_ts) >= 0) and np.all(np.diff(gt_ts) >= 0)):
                raise ValueError("Timestamps have to be sorted!")

            # remove all rows in the estimated data, where no groundtruth is present
            print(str(((np.sum(est_ts < gt_ts[0]) + np.sum(est_ts > gt_ts[len(gt_ts) - 1])) / len(
                est_ts) * 100)) + " percent of measurements countain no groundtruth, they will be removed!")
            idx_oi = [((est_ts[i] > gt_ts[0]) and (est_ts[i] < gt_ts[len(gt_ts) - 1])).tolist() for i in
                      range(len(est_ts))]
            est_ts = est_ts[idx_oi]
            self.est_pos_df = self.est_pos_df[idx_oi]

            # for each estimated point, only use the closest point in
            # TODO: make this code less complex in time
            matches = {}
            for est_ts_oi in est_ts:
                dists = [abs(est_ts_oi - val) for val in gt_ts]
                idx_oi = np.argmin(dists)
                if dists[idx_oi] < 200000000:
                    matches[est_ts_oi] = gt_ts[idx_oi]
                else:
                    print("for timestamp {}, no matching groundtruth found!".format(est_ts_oi))
                    matches[est_ts_oi] = None
            self.gt_pos_df = self.gt_pos_df[[val in matches.values() for val in self.gt_pos_df["timestamp"]]]

            if not self.gt_pos_df.shape == self.est_pos_df.shape:
                raise ValueError("Something went wrong, position dataframes of different dimension!")
        else:
            raise ValueError("No dataframes found!")

    def transform_coordinate_system(self):
        if not self.gt_pos_df.shape == self.est_pos_df.shape:
            raise ValueError("Position dataframes of different dimension! You might forgot to align the timestamps")
        s, R, t = align_umeyama(self.gt_pos_df[["p_x", "p_y", "p_z"]].to_numpy(),
                                self.est_pos_df[["p_x", "p_y", "p_z"]].to_numpy())
        print("Scaling-Factor: {}".format(str(np.round(s, 2))))
        print("Translation-Vector:")
        print(t)
        print("Rotation Matrix:")
        print(R)

        df_est_trans = np.array([s * np.dot(R, p) + t for p in self.est_pos_df[["p_x", "p_y", "p_z"]].to_numpy()])
        self.est_pos_df = pd.DataFrame({"timestamp": self.est_pos_df["timestamp"], "p_x": df_est_trans[:,0], "p_y": df_est_trans[:, 1], "p_z": df_est_trans[:, 2]})

    def get_gt_pos_df(self):
        return self.gt_pos_df

    def get_est_pos_df(self):
        return self.est_pos_df
