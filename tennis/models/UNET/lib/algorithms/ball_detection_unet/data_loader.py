import glob
import os
import random
import json

import cv2
import torch
import numpy as np

from skimage.transform import rescale, rotate

TARGET_HEIGHT = 1280
NETWORK_DIM = 720  # Inputs
NETWORK_OUTPUT_DIM = 360  # Outputs, can be different

MIN_ROTATION = -15  # Degrees
MAX_ROTATION = 15  # Degrees

class BallUNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(BallUNetDataset, self).__init__()

        # Data is stored in subdirectories. Get all of them
        self.inputs = sorted(glob.glob(data_dir + "/*/*/*_0.jpg"))
        self.targets = sorted(glob.glob(data_dir + "/*/*/*_target.json"))

        assert(len(self.inputs) == len(self.targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        try:
            # Load and convert to RGB
            stacked_images = []
            for i in range(3):
                curr_image = cv2.imread(self.inputs[idx][:-5] + f"{i}.jpg")
                curr_image = curr_image[:, :, ::-1]  # BGR to RGB
                stacked_images.append(curr_image)
            curr_input = np.concatenate(stacked_images, axis=-1)

            with open(self.targets[idx]) as f:
                target_circle = json.load(f)

            if target_circle["radius"] == 0.0:
                ball_radius = 0
            else:
                ball_radius = max(target_circle["radius"], 1) * 1.4

            # only rescale a few because it's really slow
            if np.random.uniform() < 1.0:
                original_max_dim = max(curr_input.shape[0], curr_input.shape[1])
                scale_factor_y = (TARGET_HEIGHT * random.uniform(0.8, 1.2)) / original_max_dim
                scale_factor_x = scale_factor_y * np.random.uniform(0.8, 1.2)
            else:
                scale_factor_x = 1.0
                scale_factor_y = scale_factor_x

            target = np.zeros((curr_input.shape[0], curr_input.shape[1], 1)).astype(np.uint8)

            # If ball radius is 0 then there is no ball anywhere
            if ball_radius != 0:
                cv2.circle(target, (int(target_circle["center"][0]), int(target_circle["center"][1])),
                                    round(ball_radius), color=[255, 255, 255], thickness=-1)


            # Get a random 720 crop that include the ball
            rescaled_network_dim = int(max(NETWORK_DIM / scale_factor_y, NETWORK_DIM / scale_factor_x) + 1)
            min_crop_u = max(0, target_circle["center"][0] - rescaled_network_dim * 0.75)
            max_crop_u = min(target_circle["center"][0] - rescaled_network_dim * 0.25, curr_input.shape[1] - rescaled_network_dim)
            start_crop_u = round(np.random.uniform(min_crop_u, max(min_crop_u, max_crop_u)))

            min_crop_v = max(0, target_circle["center"][1] - rescaled_network_dim * 0.75)
            max_crop_v = min(target_circle["center"][1] - rescaled_network_dim * 0.25, curr_input.shape[0] - rescaled_network_dim)
            start_crop_v = round(np.random.uniform(min_crop_v, max(min_crop_v, max_crop_v)))

            cropped_input = curr_input[start_crop_v:start_crop_v + rescaled_network_dim, start_crop_u:start_crop_u + rescaled_network_dim]
            cropped_target = target[start_crop_v:start_crop_v + rescaled_network_dim, start_crop_u:start_crop_u + rescaled_network_dim]


            # This already divides by 255
            scaled_input = rescale(cropped_input, (scale_factor_y, scale_factor_x), multichannel=True)
            scaled_target = rescale(cropped_target, (scale_factor_y, scale_factor_x), multichannel=True)

            assert(scaled_input.shape[0] == scaled_target.shape[0] and scaled_input.shape[1] == scaled_target.shape[1])

            # Check that we at least have the  network dimension shape
            if scaled_input.shape[0] < NETWORK_DIM or scaled_input.shape[1] < NETWORK_DIM:
                scaled_input = np.pad(scaled_input, (
                                                     (0, max(0, int(NETWORK_DIM - scaled_input.shape[0]))),
                                                     (0, max(0, int(NETWORK_DIM - scaled_input.shape[1]))),
                                                     (0, 0)))
                scaled_target = np.pad(scaled_target, (
                                                       (0, max(0, int(NETWORK_DIM - scaled_target.shape[0]))),
                                                       (0, max(0, int(NETWORK_DIM - scaled_target.shape[1]))),
                                                       (0, 0)))

            new_ball_location = (target_circle["center"][0] * scale_factor_x, target_circle["center"][1] * scale_factor_y)

            cropped_input = scaled_input[:NETWORK_DIM, :NETWORK_DIM]
            cropped_target = scaled_target[:NETWORK_DIM, :NETWORK_DIM]

            # Randomly rotate
            if np.random.uniform() < 0.4:
                rotation = np.random.uniform(MIN_ROTATION, MAX_ROTATION)
                cropped_input = rotate(cropped_input, rotation)
                cropped_target = rotate(cropped_target, rotation)

            # Need to resize target
            if NETWORK_DIM != NETWORK_OUTPUT_DIM:
                cropped_target = cv2.resize(cropped_target, (NETWORK_OUTPUT_DIM, NETWORK_OUTPUT_DIM))
                if len(cropped_target.shape) == 2:
                    cropped_target = np.expand_dims(cropped_target, 2)

            network_input_torch = torch.FloatTensor(cropped_input.copy()).permute(2, 0, 1)
            cropped_target_torch = torch.FloatTensor(cropped_target.copy()).permute(2, 0, 1)

            # There's a weird flip error that sometimes happens
            # Random horizontal flip
            if np.random.uniform() < 0.5:
                network_input_torch = torch.flip(network_input_torch, [-1])
                cropped_target_torch = torch.flip(cropped_target_torch, [-1])

            # We had already divided by 255, need to divide by 127.5, subtract 1
            network_input_torch = network_input_torch * (2 * np.random.uniform(0.95, 1.05)) - (1.0 * np.random.uniform(0.95, 1.05))

            return network_input_torch, cropped_target_torch

        except Exception as e:
            print("HEEYYYYYY\n")
            print(e)
            print(self.inputs[idx])
            return torch.FloatTensor(9, NETWORK_DIM, NETWORK_DIM) * 0, torch.FloatTensor(1, NETWORK_DIM, NETWORK_DIM) * 0
