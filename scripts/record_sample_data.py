from __future__ import print_function

from datetime import datetime

from robolab_turtlebot import Turtlebot, sleep

from scipy.io import savemat

import os
import cv2
import numpy as np

def main() -> None:
    # initialize turlebot
    turtle = Turtlebot(rgb=True, depth=True, pc=True)

    # sleep 2 set to receive images
    sleep(2)

    # Make new subfolder for each recording session
    folder_number = len(os.listdir("sample_data"))
    folder = os.path.join("sample_data", f"recording_{folder_number:04d}")
    os.mkdir(folder)

    i = 0
    while not turtle.is_shutting_down():
        # get K, images, and point cloud
        # Take a new set of data every now and then
        print(f"Getting record {i}")
        data = dict()
        data['K_rgb'] = turtle.get_rgb_K()
        data['K_depth'] = turtle.get_depth_K()
        data['image_rgb'] = turtle.get_rgb_image()
        data['image_depth'] = turtle.get_depth_image()
        data['point_cloud'] = turtle.get_point_cloud()

        # save data to .mat file
        filename = os.path.join(folder, f"{i:04d}.mat")
        savemat(filename, data)      

        # Save RGB image
        rgb_filepath = os.path.join(folder, f"{i:04}_rgb.png")
        cv2.imwrite(rgb_filepath, data['image_rgb'])  # BGR saved as PNG

        # Convert depth to 8-bit grayscale for visualization
        # First, convert depth to float meters
        depth_m = data['image_depth'].astype(np.float32) / 1000.0  # convert mm -> meters

        # Normalize depth for display (0-255)
        depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        # Save depth image as black-and-white
        depth_filepath = os.path.join(folder, f"{i:04d}_depth.png")
        cv2.imwrite(depth_filepath, depth_uint8)

        print("RGB and depth images saved successfully!")
        i += 1  


if __name__ == "__main__":
    main()
