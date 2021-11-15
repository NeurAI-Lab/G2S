# Copyright © NavInfo Europe 2021.

import os
from torch import utils


class GPSDataloader(utils.data.Dataset):

    def __init__(self, data_path, filenames,  altitude_with_gps=True):
        """
        Initiate GPSDataloader to load the GPS data from KITTI

        Parameters
        ----------
        data_path: Path to KITTI raw dataset (/path_to_KITTI/raw_data/sync/).
        filenames: List of filenames which are in the training set, to be used with 'get_image_path'.
        altitude_with_gps: If gps has altitude values.
        """
        super(GPSDataloader, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.altitude_with_gps = altitude_with_gps

    def txt_reader_eigen(self, path, frame_index):
        """
        Reads the GPS from the txt files

        Parameters
        ----------
        path: Path to the GPS file.
        frame_index: The frame id for the KITTI sequence.
        """
        with open(path, 'r') as f:
            poses = f.readlines()
            translations = []
            for pose in poses[frame_index - 1: frame_index + 2]:
                pose = pose.rstrip()
                if self.altitude_with_gps:
                    translation = [
                        float(pose.split(" ")[3]),
                        float(pose.split(" ")[7]),
                        float(pose.split(" ")[11])
                    ]
                else:
                    translation = [
                        float(pose.split(" ")[3]),
                        float(pose.split(" ")[11])
                    ]
                translations.append(translation)
            return translations

    def norm(self, t1, t2):
        """
        Computes the norm of the translation

        Parameters
        -----------
        t1: translation 1.
        t2: translation 2.
        """
        diff = 0
        for c1, c2 in zip(t1, t2):
            diff += (c1 - c2) ** 2
        return diff ** 0.5

    def __getitem__(self, index):
        gps = {}
        image_path = self.get_image_path(index)
        gps_path = os.path.join("/".join(image_path.split("/")[:-3]), image_path.split("/")[-4] + ".txt")
        frame_index = int(os.path.basename(image_path).split('.')[0])
        translations = self.txt_reader_eigen(gps_path, frame_index)
        gps["-1, 0"] = self.norm(translations[1], translations[0])
        gps["0, 1"] = self.norm(translations[1], translations[2])

        return gps

    def get_image_path(self, index):
        """
        To be implemented by the user to extract image path from self.filenames
        Example image path:
        /path_to_KITTI/raw_data/sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png
        """
        return ''
