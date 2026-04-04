#!/usr/bin/env python

from robolab_turtlebot import Turtlebot
import numpy as np
import cv2

x_range = (-0.3, 0.3)
z_range = (0.3, 3.0)
WINDOW = 'obstacles'


def main():

    turtle = Turtlebot(pc=True, rgb=True)
    cv2.namedWindow(WINDOW)

    while not turtle.is_shutting_down():
        # get point cloud
        turtle.wait_for_point_cloud()
        pc = turtle.get_point_cloud()
        turtle.wait_for_rgb_image()
        rgb = turtle.get_rgb_image()    # shape (480, 640, 3)

        if pc is None or rgb is None:
            continue

        # mask out floor points
        mask = pc[:, :, 1] > x_range[0]

        # mask point too far and close
        mask = np.logical_and(mask, pc[:, :, 2] > z_range[0])
        mask = np.logical_and(mask, pc[:, :, 2] < z_range[1])

        if np.count_nonzero(mask) <= 0:
            continue

        # empty image
        image = np.zeros(mask.shape)

        # assign depth i.e. distance to image
        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)
        im_color = cv2.applyColorMap(255 - image.astype(np.uint8),
                                     cv2.COLORMAP_JET)

        # OPTIONAL: ensure both images have same type
        rgb_uint8 = rgb.astype(np.uint8)

        # combine images side by side
        combined = np.hstack((rgb_uint8, im_color))

        # show image
        cv2.imshow(WINDOW, combined)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()