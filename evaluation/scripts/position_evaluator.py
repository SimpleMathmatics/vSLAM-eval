import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition as sd
import os
import pandas as pd
import numpy as np
from json_helper import JsonHelper


class Evaluator:
    def __init__(self, gt_df, est_df):
        self.gt_df = gt_df
        self.est_df = est_df

    def create_pos_dif_plots(self, outdir):
        def create_pos_dif_plots(df1, df2, col1, col2):
            fig1 = px.line(df1, x=col1, y=col2)
            fig2 = px.line(df2, x=col1, y=col2)
            fig2.update_traces(line=dict(color='rgba(100,0,0,0.2)'))
            fig3 = go.Figure(data=fig1.data + fig2.data)
            return fig3

        # create plot directory, if not present
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # x against y
        fig = create_pos_dif_plots(self.est_df, self.gt_df, "p_x", "p_y")
        fig.write_image(os.path.join(outdir, "x_y.png"), engine="kaleido")

        # x against z
        fig = create_pos_dif_plots(self.est_df, self.gt_df, "p_x", "p_z")
        fig.write_image(os.path.join(outdir, "x_z.png"), engine="kaleido")

        # y against z
        fig = create_pos_dif_plots(self.est_df, self.gt_df, "p_y", "p_z")
        fig.write_image(os.path.join(outdir, "y_z.png"), engine="kaleido")

        # transform via pca and plot pc1 against pc2
        pca1 = sd.PCA(n_components=2)
        pca1.fit(self.est_df[["p_x", "p_y", "p_z"]])
        transformed_est_df = pca1.transform(self.est_df[["p_x", "p_y", "p_z"]])
        transformed_est_df = pd.DataFrame({"PC1": transformed_est_df[:, 0], "PC2": transformed_est_df[:, 1]})
        pca2 = sd.PCA(n_components=2)
        pca2.fit(self.gt_df[["p_x", "p_y", "p_z"]])
        transformed_gt_df = pca2.transform(self.gt_df[["p_x", "p_y", "p_z"]])
        transformed_gt_df = pd.DataFrame({"PC1": transformed_gt_df[:, 0], "PC2": transformed_gt_df[:, 1]})
        fig = create_pos_dif_plots(transformed_est_df, transformed_gt_df, "PC1", "PC2")
        fig.write_image(os.path.join(outdir, "pca.png"), engine="kaleido")

    def calculate_diff(self, outdir_plot, outdir_json):
        dists = []
        for i in range(self.est_df.shape[0]):
            p1 = np.array([self.est_df.iloc[i, 1], self.est_df.iloc[i, 2], self.est_df.iloc[i, 3]])
            p2 = np.array([self.gt_df.iloc[i, 1], self.gt_df.iloc[i, 2], self.gt_df.iloc[i, 3]])
            dist = np.linalg.norm(p1-p2)
            dists.append(dist)
        mean_diff = np.mean(dists)
        print("mean difference position error: {}".format(mean_diff))

        # generate plot of the distances of each timeunit
        fig = go.Figure(data=go.Scatter(x=self.est_df.timestamp.to_numpy(), y=dists))
        fig.write_image(os.path.join(outdir_plot, "distance.png"), engine="kaleido")

        # generate json file with result and write to
        jh = JsonHelper()
        jh.add_json(os.path.join(outdir_json, "results.txt"), "mean distance error", mean_diff)




