import os


class CommandGenerator:
    @staticmethod
    def orb(dataset, path_to_orb, path_to_data, filename):
        sep = " "
        if dataset == "euroc":
            ORB_voc = os.path.join(path_to_orb, "Vocabulary", "ORBvoc.txt")
            cam_config = os.path.join(path_to_orb, "Examples", "Monocular", "EuRoC.yaml")
            timestamps = os.path.join(path_to_orb, "Examples", "Monocular", "EuRoC_TimeStamps", filename + ".txt")
            init_cmd = os.path.join(path_to_orb, "Examples", "Monocular", "mono_euroc")
            path_to_data_plus_file = os.path.join(path_to_data, filename)
            command = init_cmd + sep + ORB_voc + sep + cam_config + sep + path_to_data_plus_file + sep + timestamps
        else:
            command = None

        return command




