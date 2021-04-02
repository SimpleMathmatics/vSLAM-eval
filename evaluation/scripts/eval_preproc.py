import pandas as pd
import numpy as np
from align import align_umeyama
import os


class FramePreprocessor:
    def __init__(self, gt_filepath, est_filepath, dataset_type, outdir_data):
        self.dataset_type = dataset_type
        self.gt_filepath = gt_filepath
        self.est_filepath = est_filepath
        self.outdir_data = outdir_data
        self.gt_pos_df = None
        self.est_pos_df = None

    @staticmethod
    def standardize_timestamp(ts):
        ts = '{0:.10f}'.format(ts)
        ts_head = ts[0:10]
        if ts[10] == ".":
            ts_tail = ts[11:14]
        else:
            ts_tail = ts[10:13]
        ts_out = ts_head + "." + ts_tail
        return float(ts_out)

    def create_gt_pos_df(self):
        if self.dataset_type == "euroc":
            gt_df = pd.read_csv(self.gt_filepath, sep=",")
            
            # for now only use positional dataset_type
            gt_pos_df = pd.DataFrame({"timestamp": gt_df["#timestamp"],
                                      "p_x": gt_df[" p_RS_R_x [m]"],
                                      "p_y": gt_df[" p_RS_R_y [m]"],
                                      "p_z": gt_df[" p_RS_R_z [m]"]})
            gt_pos_df.timestamp = gt_pos_df.timestamp.apply(self.__class__.standardize_timestamp)
            self.gt_pos_df = gt_pos_df
            
    def create_est_pos_df(self):
        def standardize_seps(filepath, target_sep, invalid_seps):
            lines_out = []
            file = open(filepath, "r+")
            lines = file.readlines()
            for line in lines:
                for invalid_sep in invalid_seps:
                    line = line.replace(invalid_sep, target_sep)
                lines_out.append(line)
            file.truncate(0)
            file.close()
            file = open(filepath, "w")
            file.writelines(lines_out)
            file.close()

        standardize_seps(invalid_seps=["    ", "   ", "  "],
                         target_sep=" ",
                         filepath=self.est_filepath)

        if self.dataset_type == "euroc":
            df_calc = pd.read_csv(self.est_filepath,
                                  sep=" ",
                                  header=None,
                                  names=["timestamp", "p_x", "p_y", "p_z", "1", "2", "3", "4"])
            df_calc.timestamp = df_calc.timestamp.apply(self.__class__.standardize_timestamp)
            self.est_pos_df = df_calc[["timestamp", "p_x", "p_y", "p_z"]]

    def align_timestamps(self):
        if all([isinstance(self.est_pos_df, pd.DataFrame), isinstance(self.gt_pos_df, pd.DataFrame)]):
            gt_ts = self.gt_pos_df["timestamp"].to_numpy()
            est_ts = self.est_pos_df["timestamp"].to_numpy()
            if not (np.all(np.diff(est_ts) >= 0) and np.all(np.diff(gt_ts) >= 0)):
                raise ValueError("Timestamps have to be sorted!")

            # remove all rows in the estimated data, where no groundtruth is present
            perc_no_gt = ((np.sum(est_ts < gt_ts[0]) + np.sum(est_ts > gt_ts[len(gt_ts) - 1])) / len(
                est_ts) * 100)
            if perc_no_gt != 0:

                print(str(perc_no_gt) + " percent of measurements countain no groundtruth, they will be removed!")
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

    def transform_coordinate_system(self, outdir):
        if not self.gt_pos_df.shape == self.est_pos_df.shape:
            raise ValueError("Position dataframes of different dimension! You might forgot to align the timestamps")
        s, R, t = align_umeyama(self.gt_pos_df[["p_x", "p_y", "p_z"]].to_numpy(),
                                self.est_pos_df[["p_x", "p_y", "p_z"]].to_numpy())
        print("Scaling-Factor: {}".format(str(np.round(s, 2))))
        np.save(os.path.join(outdir, "scale.npy"),np.array(s))
        print("Translation-Vector:")
        print(t)
        np.save(os.path.join(outdir, "trans_vec.npy"), t)
        print("Rotation Matrix:")
        print(R)
        np.save(os.path.join(outdir, "rotation_matrix.npy"), R)

        df_est_trans = np.array([s * np.dot(R, p) + t for p in self.est_pos_df[["p_x", "p_y", "p_z"]].to_numpy()])
        self.est_pos_df = pd.DataFrame({"timestamp": self.est_pos_df["timestamp"], "p_x": df_est_trans[:, 0], "p_y": df_est_trans[:, 1], "p_z": df_est_trans[:, 2]})
        self.est_pos_df.to_csv(os.path.join(outdir, "est_df_transformed.csv"))

    def get_gt_pos_df(self):
        return self.gt_pos_df

    def get_est_pos_df(self):
        return self.est_pos_df
