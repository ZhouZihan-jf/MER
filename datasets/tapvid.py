from glob import glob
import numpy as np
import torch
import torch.utils.data as data
import pickle
import os
from typing import Tuple, Optional
from .tapvid_evaluation_datasets import *


class TapVidDAVIS(data.Dataset):
    def __init__(self,
                 davis_points_path: str,
                 query_mode: str = 'strided',
                 full_resolution=False,
                 resolution: Optional[Tuple[int, int]] = (256, 256)):
        self.davis_points_path = davis_points_path
        self.query_mode = query_mode
        self.full_resolution = full_resolution
        self.resolution = resolution
        
        self.load_pkls()


    def load_pkls(self):
        samples = glob(os.path.join(self.davis_points_path, '*.pkl'))
        
        self.davis_points_dataset = []
        for sample in samples:
            with open(sample, 'rb') as f:
                data = pickle.load(f)
                self.davis_points_dataset.append(data)
                

    def __len__(self):
        if self.full_resolution:
            raise NotImplementedError("Full resolution mode is not implemented.")
        else:
            return len(self.davis_points_dataset)

    def __getitem__(self, idx):
        data = self.prepare_data(idx)
        rgbs, query_points, trajectories, visibilities  = TapVidDAVIS.preprocess_dataset_element(data)
        return rgbs, query_points, trajectories, visibilities

        
    def prepare_data(self, idx):
        if self.full_resolution:
            raise NotImplementedError("Full resolution mode is not implemented.")
        else:
            frames = self.davis_points_dataset[idx]['video']
            target_points = self.davis_points_dataset[idx]['points']
            target_occ = self.davis_points_dataset[idx]['occluded']

            if self.resolution is not None and self.resolution != frames.shape[1:3]:
                frames = resize_video(frames, self.resolution)

            frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

            if self.query_mode == 'strided':
                converted = sample_queries_strided(target_occ, target_points, frames)
            elif self.query_mode == 'first':
                converted = sample_queries_first(target_occ, target_points, frames)
            else:
                raise ValueError(f'Unknown query mode {self.query_mode}.')

        return converted
    
    @staticmethod
    def preprocess_dataset_element(dataset_element):
        rgbs = torch.from_numpy(dataset_element['video']).permute(0, 1, 4, 2, 3)
        query_points = torch.from_numpy(dataset_element['query_points'])
        trajectories = torch.from_numpy(dataset_element['target_points']).permute(0, 2, 1, 3)
        visibilities = ~torch.from_numpy(dataset_element['occluded']).permute(0, 2, 1)

        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        # Convert query points from (t, y, x) to (t, x, y)
        query_points = query_points[:, :, [0, 2, 1]]

        # Ad hoc fix for Kubric reporting invisible query points when close to the crop boundary, e.g., x=110, y=-1e-5
        for point_idx in range(n_points):
            query_point = query_points[0, point_idx]
            query_visible = visibilities[0, query_point[0].long(), point_idx]
            if query_visible:
                continue

            x, y = query_point[1:]
            x_at_boundary = min(abs(x - 0), abs(x - (width - 1))) < 1e-3
            y_at_boundary = min(abs(y - 0), abs(y - (height - 1))) < 1e-3
            x_inside_window = 0 <= x <= width - 1
            y_inside_window = 0 <= y <= height - 1

            if x_at_boundary and y_inside_window or x_inside_window and y_at_boundary or x_at_boundary and y_at_boundary:
                visibilities[0, query_point[0].long(), point_idx] = 1

        # Check dimensions are correct
        assert batch_size == 1
        assert rgbs.shape == (batch_size, n_frames, channels, height, width)
        assert query_points.shape == (batch_size, n_points, 3)
        assert trajectories.shape == (batch_size, n_frames, n_points, 2)
        assert visibilities.shape == (batch_size, n_frames, n_points)

        # Check that query points are visible
        assert torch.all(visibilities[0, query_points[0, :, 0].long(), torch.arange(n_points)] == 1), \
            "Query points must be visible"

        # Check that query points are correct
        assert torch.allclose(
            query_points[0, :, 1:].float(),
            trajectories[0, query_points[0, :, 0].long(), torch.arange(n_points)].float(),
            atol=1.0,
        )
        
        return rgbs[0], query_points[0], trajectories[0], visibilities[0]
        

if __name__ == "__main__":
    td = TapVidDAVIS('/dataset/zzh/tapvid_davis')
    print(td[0].size())
        