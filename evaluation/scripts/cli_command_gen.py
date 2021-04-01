import os


class CommandGenerator:
    @staticmethod
    def orb(dataset, path_to_orb, path_to_data, path_to_config, filename, resolution):
        sep = " "
        if dataset == "euroc":
            ORB_voc = os.path.join(path_to_orb, "Vocabulary", "ORBvoc.txt")
            if resolution == 1:
                cam_config = os.path.join(path_to_config, "EuRoC.yaml")
            else:
                cam_config = os.path.join(path_to_config, "EuRoC_0" + str(resolution)[2] + ".yaml")
            timestamps = os.path.join(path_to_orb, "Examples", "Monocular", "EuRoC_TimeStamps", filename + ".txt")
            init_cmd = os.path.join(path_to_orb, "Examples", "Monocular", "mono_euroc")
            path_to_data_plus_file = os.path.join(path_to_data, filename)
            command = init_cmd + sep + ORB_voc + sep + cam_config + sep + path_to_data_plus_file + sep + timestamps
        else:
            command = None
        return command

    @staticmethod
    def dsm(dataset, path_to_dsm, path_to_data, path_to_config, filename, resolution, path_to_orb):
        sep = " "
        if dataset == "euroc":
            init_cmd = os.path.join(path_to_dsm, "build", "bin", "EurocExample")
            data = os.path.join(path_to_data, filename, "mav0", "cam0", "data")
            settings = os.path.join(path_to_dsm, "Examples", "EurocData", "settings.txt")
            timestamps = os.path.join(path_to_orb, "Examples", "Monocular", "EuRoC_TimeStamps", filename + ".txt")
            if resolution < 1:
                calibration = os.path.join(path_to_config, "calib_dsm_0" + str(resolution)[2] + ".txt")
            else:
                calibration = os.path.join(path_to_config, "calib_dsm.txt")
            command = init_cmd + sep + data + sep + timestamps + sep + calibration + sep + settings + sep + "autorun"
        else:
            command = None
        return command





