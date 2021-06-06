import pptk
import pandas as pd
from plyfile import PlyData
import numpy as np
from scipy.spatial.distance import cdist
import plotly.graph_objects as go

class PointCloud:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.color_mat = None

    def from_file(self, filepath):
        if filepath.__contains__(".ply"):
            plydata = PlyData.read(filepath)
            self.x = plydata.elements[0].data['x']
            self.y = plydata.elements[0].data['y']
            self.z = plydata.elements[0].data['z']
        else:
            try:
                df = pd.read_csv(filepath, sep=" ")
                self.x = df.x
                self.y = df.y
                self.z = df.z
            except:
                raise ValueError("Error when reading in point-cloud-file!")

    def generate_color_mat(self, color):
        array_1 = np.array([1 for _ in range(len(self.x))])
        if color == "red":
            color_mat = np.vstack((array_1, array_1 * 0, array_1 * 0)).transpose()
        elif color == "green":
            color_mat = np.vstack((array_1 * 0, array_1, array_1 * 0)).transpose()
        elif color == "blue":
            color_mat = np.vstack((array_1 * 0, array_1 * 0, array_1)).transpose()
        elif color == "white":
            color_mat = np.vstack((array_1, array_1, array_1)).transpose()
        else:
            raise ValueError("Color can be only green, blue or red!")
        self.color_mat = color_mat

    def from_vector(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def point_mat(self):
        return np.vstack((self.x, self.y, self.z)).transpose()

    @property
    def n(self):
        return len(self.x)

    def transform_points(self, rot_mat, trans, scale):
        point_mat_trans = np.array([scale * np.dot(rot_mat, p) + trans for p in self.point_mat])
        self.x = point_mat_trans[:, 0]
        self.y = point_mat_trans[:, 1]
        self.z = point_mat_trans[:, 2]

    def add_point_cloud(self, second_point_cloud):
        self.x = np.append(self.x, second_point_cloud.x)
        self.y = np.append(self.y, second_point_cloud.y)
        self.z = np.append(self.z, second_point_cloud.z)
        self.color_mat = np.concatenate((self.color_mat, second_point_cloud.color_mat), axis=0)
        print(self.color_mat.shape)
        print(self.color_mat)
        print(self.color_mat[1, 1])
        print(self.color_mat[self.color_mat.shape[0]-1, self.color_mat.shape[1]-1])

    def visualize_point_cloud(self, point_size=None):
        v = pptk.viewer(self.point_mat)

        if type(self.color_mat) is np.ndarray:
            v.attributes(self.color_mat)

        if point_size:
            v.set(point_size=point_size)

    def compare_to_gt(self, gt_point_cloud):
        # get 1000 random points from point cloud
        random_indices = np.random.choice(self.n, size=500, replace=True)
        eval_points = self.point_mat[random_indices, :]
        gt_points = gt_point_cloud.point_mat
        dists = []
        i = 0
        for p in eval_points:
            dists_to_p = cdist(p.reshape(1, 3), gt_points)
            min_dist = min(dists_to_p[0])
            dists.append(min_dist)
            i += 1
            if i % 100 == 0:
                print("Calculated {} percent of the distances".format(str(np.round(i/5))))
        return np.array(dists)


if __name__ == "__main__":
    if False:
        # read in evaluated point cloud
        pq_eval = PointCloud()
        pq_eval.from_file("../results/V101/ORB/data/PointCloud.txt")
        pq_eval.generate_color_mat("red")
        pq_eval.transform_points(trans=np.load("../results/V101/ORB/data/trans_vec.npy"),
                                 rot_mat=np.load("../results/V101/ORB/data/rotation_matrix.npy"),
                                 scale=np.load("../results/V101/ORB/data/scale.npy"))

        # read in ground truth point cloud
        pq_gt = PointCloud()
        pq_gt.from_file('../results/V101/ORB/data/PointCloud_gt.ply')
        pq_gt.generate_color_mat("white")
        # pq_gt.visualize_point_cloud()

        # dists = pq_eval.compare_to_gt(pq_gt)
        # np.save("../results/V101/DSM/data/distances_pointcloud.npy", dists)
        pq_eval.add_point_cloud(second_point_cloud=pq_gt)
        pq_eval.visualize_point_cloud(point_size=0.003)

    if False:
        dists_dsm = np.load("../results/V101/DSM/data/distances_pointcloud.npy")
        dists_orb = np.load("../results/V101/ORB/data/distances_pointcloud.npy")
        print(np.mean(dists_dsm))
        print(np.mean(dists_orb))
        print(max(dists_dsm))
        print(max(dists_orb))
        import plotly.figure_factory as ff
        import numpy as np

        hist_data = [dists_dsm, dists_orb]

        # Create distplot with curve_type set to 'normal'
        fig = ff.create_distplot(hist_data, show_hist=False, group_labels=["DSM", "ORB"])

        # Add title
        fig.update_layout(title_text='Distance from evaluated point to ground truth')
        fig.show()

    if False:
        # read in evaluated point cloud
        pq_eval = PointCloud()
        pq_eval.from_file("C:/Users/julia/Downloads/PointCloud.txt")
        pq_eval.transform_points(rot_mat=np.load("../results/V101/ORB/data/rotation_matrix.npy"),
                                 scale=np.load("../results/V101/ORB/data/scale.npy"),
                                 trans=np.load("../results/V101/ORB/data/trans_vec.npy"))
        pq_eval.generate_color_mat("red")

        # read in ground truth point cloud
        pq_gt = PointCloud()
        pq_gt.from_file('C:/Users/julia/Downloads/data.ply')
        pq_gt.generate_color_mat("white")

        # dists = pq_eval.compare_to_gt(pq_gt)
        # np.save("../results/V101/DSM/data/distances_pointcloud.npy", dists)
        # pq_eval.add_point_cloud(second_point_cloud=pq_gt)
        dists = pq_eval.compare_to_gt(pq_gt)
        np.save("../results/V101/ORB/data/distances_pointcloud.npy", dists)

    if False:
        import os
        filenames = pd.read_csv("../config/data_meta.csv", sep=";").filename.tolist()

        dataset_paths = [os.path.join("..", "results", filename) for filename in filenames]

        result_paths_orb = [os.path.join(data_path, "ORB", "data") for data_path in dataset_paths]
        result_paths_dsm = [os.path.join(data_path, "DSM", "data") for data_path in dataset_paths]
        result_paths_dso = [os.path.join(data_path, "DSO", "data") for data_path in dataset_paths]

        n_orb = []
        n_dsm = []
        n_dso = []

        point_dist_orb = []
        point_dist_dsm = []
        point_dist_dso = []

        all_dists_orb = []
        all_dists_dsm = []
        all_dists_dso = []

        for i in range(len(result_paths_dso)):
            # orb
            pq_orb = PointCloud()
            pq_orb.from_file(os.path.join(result_paths_orb[i], "PointCloud.txt"))
            n_orb.append(pq_orb.n)
            pq_orb.transform_points(
                rot_mat=np.load(os.path.join(os.path.join(result_paths_orb[i], "rotation_matrix.npy"))),
                trans=np.load(os.path.join(os.path.join(result_paths_orb[i], "trans_vec.npy"))),
                scale=np.load(os.path.join(os.path.join(result_paths_orb[i], "scale.npy")))
            )

            # dsm
            pq_dsm = PointCloud()
            pq_dsm.from_file(os.path.join(result_paths_dsm[i], "PointCloud.ply"))
            n_dsm.append(pq_dsm.n)
            pq_dsm.transform_points(
                rot_mat=np.load(os.path.join(os.path.join(result_paths_dsm[i], "rotation_matrix.npy"))),
                trans=np.load(os.path.join(os.path.join(result_paths_dsm[i], "trans_vec.npy"))),
                scale=np.load(os.path.join(os.path.join(result_paths_dsm[i], "scale.npy")))
            )

            # dso
            pq_dso = PointCloud()
            pq_dso.from_file(os.path.join(result_paths_dso[i], "PointCloud.txt"))
            n_dso.append(pq_dso.n)

            pq_dso.transform_points(
                rot_mat=np.load(os.path.join(os.path.join(result_paths_dso[i], "rotation_matrix.npy"))),
                trans=np.load(os.path.join(os.path.join(result_paths_dso[i], "trans_vec.npy"))),
                scale=np.load(os.path.join(os.path.join(result_paths_dso[i], "scale.npy")))
            )

            if i not in range(5):
                pq_gt = PointCloud()
                pq_gt.from_file(os.path.join(result_paths_orb[i], "PointCloud_gt.ply"))

                dists_orb = pq_orb.compare_to_gt(pq_gt)
                dists_dsm = pq_dsm.compare_to_gt(pq_gt)
                dists_dso = pq_dso.compare_to_gt(pq_gt)

                point_dist_orb.append(np.mean(dists_orb))
                point_dist_dsm.append(np.mean(dists_dsm))
                point_dist_dso.append(np.mean(dists_dso))

                all_dists_dso.append(dists_dso)
                all_dists_dsm.append(dists_dsm)
                all_dists_orb.append(dists_orb)

        all_dists_dso = np.concatenate(all_dists_dso)
        all_dists_dsm = np.concatenate(all_dists_dsm)
        all_dists_orb = np.concatenate(all_dists_orb)
        
        plot = go.Figure(layout_yaxis_range=[0, 0.8])
        plot.add_trace(go.Box(y=all_dists_orb, name="ORB"))
        plot.add_trace(go.Box(y=all_dists_dsm, name="DSM"))
        plot.add_trace(go.Box(y=all_dists_dso, name="DSO"))
        plot.update_layout(showlegend=False)
        plot.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Euclidean distance to true position"
        )
        plot.show()

        print(point_dist_dso)
        print(point_dist_dsm)
        print(point_dist_orb)

        print(n_dsm)
        print(n_dso)
        print(n_orb)







