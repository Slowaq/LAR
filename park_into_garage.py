import cv2
from robolab_turtlebot import Turtlebot
import numpy as np

# get senzor data
# compute centre point
# keep parking until distance is small +- 25 cm from camera


def main():
    print("main started")
    turtle = Turtlebot(rgb=True, pc=True)
    turtle.wait_for_rgb_image()

    response = 0.002

    cv2.namedWindow("camera")
    cv2.namedWindow("depth")

    print("window created")
    
    inGarage = False
    centre = None
    
    while not turtle.is_shutting_down() and inGarage == False:

        # while centre is None:
        #     frame = turtle.get_rgb_image()
        #     if frame is None:
        #         continue
        # centre = find_garage(frame)
        
        # --- POINT CLOUD ---
        pc = turtle.get_point_cloud()
        if pc is None:
            print("Pointcloud is None")
            continue

        image = np.zeros(pc.shape[:2])

        mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

        depth_vis = cv2.applyColorMap(
            255 - image.astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        




def find_garage(frame):
    # --- PURPLE DETECTION ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_purple = np.array([100, 90, 50])
    upper_purple = np.array([140, 160, 255])
    
    #lower_purple = np.array([125, 80, 80])
    #upper_purple = np.array([160, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # keep vertical shapes
        if h > w:
            edges.append((x, y, w, h))

    # sort left to right
    edges = sorted(edges, key=lambda e: e[0])

    # --- FIND CENTER BETWEEN EDGES ---
    if len(edges) >= 2:
        left = edges[0]
        right = edges[-1]

        lx = left[0] + left[2] // 2
        rx = right[0] + right[2] // 2

        # middle point
        cx = (lx + rx) // 2
        cy = frame.shape[0] // 2

        # draw edges
        cv2.rectangle(frame, (left[0], left[1]),
                      (left[0]+left[2], left[1]+left[3]), (255, 0, 255), 2)

        cv2.rectangle(frame, (right[0], right[1]),
                      (right[0]+right[2], right[1]+right[3]), (255, 0, 255), 2)

        # draw middle point (RED)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(frame, f"Center ({cx})",
                    (cx-40, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)
        
        width = frame.shape[1]
        center_x = width // 2
        error = cx - center_x
    
    return error, [cx, cy]
    