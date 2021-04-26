import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition as sd
import os
import pandas as pd
import numpy as np
from eval_preproc import FramePreprocessor


class Evaluator:
    def __init__(self, gt_df_orb, gt_df_dsm, gt_df_dso, est_df_orb, est_df_dsm, est_df_dso):
        self.gt_df_orb = gt_df_orb
        self.gt_df_dsm = gt_df_dsm
        self.gt_df_dso = gt_df_dso
        self.est_df_orb = est_df_orb
        self.est_df_dsm = est_df_dsm
        self.est_df_dso = est_df_dso

    @staticmethod
    def create_line_plots(df_orb, df_dsm, df_dso, col1, col2, xlabel,
                          ylabel, plot_gt=True, df_gt=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_dsm[col1], y=df_dsm[col2],
                                 mode='lines',
                                 name='DSM'))
        fig.add_trace(go.Scatter(x=df_orb[col1], y=df_orb[col2],
                                 mode='lines',
                                 name='ORB'))
        fig.add_trace(go.Scatter(x=df_dso[col1], y=df_dso[col2],
                                 mode='lines',
                                 name='DSO'))

        if plot_gt:
            fig.add_trace(go.Scatter(x=df_gt[col1], y=df_gt[col2],
                                     mode='lines',
                                     name='Groundtruth'))
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel
            )
        return fig

    def create_pos_dif_plots(self):
        # transform via pca and plot pc1 against pc2
        pca1 = sd.PCA(n_components=2)
        pca1.fit(self.gt_df_orb[["p_x", "p_y", "p_z"]])
        transformed_gt_df = pca1.transform(self.gt_df_orb[["p_x", "p_y", "p_z"]])
        transformed_gt_df = pd.DataFrame({"PC1": transformed_gt_df[:, 0], "PC2": transformed_gt_df[:, 1]})

        pca2 = sd.PCA(n_components=2)
        pca2.fit(self.est_df_orb[["p_x", "p_y", "p_z"]])
        transformed_orb_df = pca2.transform(self.est_df_orb[["p_x", "p_y", "p_z"]])
        transformed_orb_df = pd.DataFrame({"PC1": transformed_orb_df[:, 0], "PC2": transformed_orb_df[:, 1]})

        pca2 = sd.PCA(n_components=2)
        pca2.fit(self.est_df_dsm[["p_x", "p_y", "p_z"]])
        transformed_dsm_df = pca2.transform(self.est_df_dsm[["p_x", "p_y", "p_z"]])
        transformed_dsm_df = pd.DataFrame({"PC1": transformed_dsm_df[:, 0], "PC2": transformed_dsm_df[:, 1]})

        pca2 = sd.PCA(n_components=2)
        pca2.fit(self.est_df_dso[["p_x", "p_y", "p_z"]])
        transformed_dso_df = pca2.transform(self.est_df_dso[["p_x", "p_y", "p_z"]])
        transformed_dso_df = pd.DataFrame({"PC1": transformed_dso_df[:, 0], "PC2": transformed_dso_df[:, 1]})

        fig = self.__class__.create_line_plots(df_gt=transformed_gt_df, df_dsm=transformed_dsm_df,
                                               df_dso=transformed_dso_df, df_orb=transformed_orb_df,
                                               col1="PC1", col2="PC2", xlabel="PC1", ylabel="PC2")
        return fig

    def plot_diff(self):
        dists_orb = self.__class__.calculate_diff(self.gt_df_orb, self.est_df_orb)
        dists_dsm = self.__class__.calculate_diff(self.gt_df_dsm, self.est_df_dsm)
        dists_dso = self.__class__.calculate_diff(self.gt_df_dso, self.est_df_dso)

        fig = self.__class__.create_line_plots(df_dso=dists_dso,
                                               df_dsm=dists_dsm,
                                               df_orb=dists_orb,
                                               plot_gt=False,
                                               col1="timestamp",
                                               col2="dist",
                                               xlabel="time in s",
                                               ylabel="position error in m")
        return fig

    @staticmethod
    def calculate_diff(gt_df, est_df):
        dists = []
        gt_df = gt_df[["timestamp", "p_x", "p_y", "p_z"]]
        est_df = est_df[["timestamp", "p_x", "p_y", "p_z"]]
        for i in range(gt_df.shape[0]):
            p1 = np.array([est_df.iloc[i, 1], est_df.iloc[i, 2], est_df.iloc[i, 3]])
            p2 = np.array([gt_df.iloc[i, 1], gt_df.iloc[i, 2], gt_df.iloc[i, 3]])
            dist = np.linalg.norm(p1-p2)
            dists.append(dist)
        dist_df = pd.DataFrame({"timestamp": gt_df.timestamp - gt_df.timestamp[0],
                                "dist": np.array(dists)})

        return dist_df


if __name__ == "__main__":
    if True:
        preproc = FramePreprocessor(gt_filepath=os.path.join("../results/V102/DSM/data/position_gt.csv"),
                                    est_filepath=os.path.join("../results/V102/DSM/data/est_df_transformed.csv"),
                                    dataset_type="euroc",
                                    outdir_data=None)
        df = pd.read_csv(os.path.join("../results/V102/DSM/data/est_df_transformed.csv"),
                                      index_col=False)
        df = df.drop(df.columns[0], axis=1)
        preproc.est_pos_df = df
        preproc.create_gt_pos_df()
        print(preproc.get_gt_pos_df().head())
        print(preproc.get_gt_pos_df().shape)
        print(preproc.est_pos_df.head())
        print(preproc.est_pos_df.shape)
        preproc.align_timestamps()
        preproc.get_gt_pos_df().to_csv("../results/V102/DSM/data/position_gt.csv", index=False)

        preproc = FramePreprocessor(gt_filepath=os.path.join("../results/V102/DSO/data/position_gt.csv"),
                                    est_filepath=os.path.join("../results/V102/DSO/data/est_df_transformed.csv"),
                                    dataset_type="euroc",
                                    outdir_data=None)
        df = pd.read_csv(os.path.join("../results/V102/DSO/data/est_df_transformed.csv"),
                                      index_col=False)
        df = df.drop(df.columns[0], axis=1)
        preproc.est_pos_df = df
        preproc.create_gt_pos_df()
        print(preproc.get_gt_pos_df().head())
        print(preproc.get_gt_pos_df().shape)
        print(preproc.est_pos_df.head())
        print(preproc.est_pos_df.shape)
        preproc.align_timestamps()
        preproc.get_gt_pos_df().to_csv("../results/V102/DSO/data/position_gt.csv", index=False)

        preproc = FramePreprocessor(gt_filepath=os.path.join("../results/V102/ORB/data/position_gt.csv"),
                                    est_filepath=os.path.join("../results/V102/ORB/data/est_df_transformed.csv"),
                                    dataset_type="euroc",
                                    outdir_data=None)
        df = pd.read_csv(os.path.join("../results/V102/ORB/data/est_df_transformed.csv"),
                                      index_col=False)
        df = df.drop(df.columns[0], axis=1)
        preproc.est_pos_df = df
        preproc.create_gt_pos_df()
        print(preproc.get_gt_pos_df().head())
        print(preproc.get_gt_pos_df().shape)
        print(preproc.est_pos_df.head())
        print(preproc.est_pos_df.shape)
        preproc.align_timestamps()
        preproc.get_gt_pos_df().to_csv("../results/V102/ORB/data/position_gt.csv", index=False)

    ev = Evaluator(gt_df_orb=pd.read_csv("../results/V102/ORB/data/position_gt.csv"),
                   gt_df_dsm=pd.read_csv("../results/V102/DSM/data/position_gt.csv"),
                   gt_df_dso=pd.read_csv("../results/V102/DSO/data/position_gt.csv"),
                   est_df_dsm=pd.read_csv("../results/V102/DSM/data/est_df_transformed.csv"),
                   est_df_orb=pd.read_csv("../results/V102/ORB/data/est_df_transformed.csv"),
                   est_df_dso=pd.read_csv("../results/V102/DSO/data/est_df_transformed.csv"))
    ev.plot_diff().show()




