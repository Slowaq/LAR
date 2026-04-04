from robolab_turtlebot import Turtlebot, Rate
import numpy as np
import cv2
from typing import Optional, Tuple

def main():
    turtle = Turtlebot(rgb=True, pc=True)
    turtle.wait_for_rgb_image()
    turtle.wait_for_point_cloud()

    RESPONSE = 0.003
    target_distance = 0.5  # 50 cm
    stoping = False

    # State dictionary to hold the current frame for the mouse callback
    state = {"frame_bgr": None}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_bgr = param["frame_bgr"]
            if frame_bgr is not None:
                h, w = frame_bgr.shape[:2]
                # Check if the click is within the left image (the original frame, not the mask)
                if x < w and y < h:
                    pixel_bgr = frame_bgr[y, x]
                    # Convert the 1x1 BGR pixel to HSV
                    pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                    print(f"Clicked pixel (x={x}, y={y}) -> BGR: {pixel_bgr}, HSV: {pixel_hsv}")
                else:
                    print("Clicked outside the original frame (on the mask).")

    # Create the window and attach the callback before the loop begins
    cv2.namedWindow("combined")
    cv2.setMouseCallback("combined", mouse_callback, state)

    while not turtle.is_shutting_down() and not stoping:
        # --- RGB OBRAZ ---
        frame = turtle.get_rgb_image() 
        if frame is None:
            print("No RGB image")
            continue

        # --- POINT CLOUD ---
        pc = turtle.get_point_cloud()
        if pc is None:
            print("Pointcloud is None")
            continue

        linear = 0.0
        angular = 0.0
                
        # Unpack the new signature from find_pylon
        coords, frame_bgr, mask = find_pylon(frame)
        
        # Store a copy of the frame so the mouse callback can read the colors 
        # accurately, even if we draw on it later
        state["frame_bgr"] = frame_bgr.copy()

        # Calculate error and extract x, y since find_pylon no longer returns error
        if coords is not None:
            x, y = coords
            error = x - (frame_bgr.shape[1] // 2 + 50)
        else:
            x, y = None, None
            error = None
                
        if error is None:
            angular = 0.3  # hľadanie objektu
        else:
            angular = -RESPONSE * error
            
        # --- ZÍSKANIE VZDIALENOSTI Z POINT CLOUDU ---
        if x is not None and y is not None: 
            distance = pc[y, x, 2]  # Z = vzdialenosť dopredu, pc.shape = (480, 640, 3)
        else:
            distance = None

        # ak máme validnú vzdialenosť
        if distance is not None and not np.isnan(distance):
            # riadenie dopredného pohybu    
            # print(f"distance: {distance:.2f}, target_distace: {target_distance:.2f}")
            if distance > target_distance:
                if abs(error) < 100:
                    # print("Going after target")
                    linear = 0.1    # jedem dopredu
                else:
                    # print("Angular error is too large - turning on the spot")
                    linear = 0.0    # jenom otaceni
            else:
                # print("Close enough to the target - stopping")

                linear = 0.0  # zastav
                # stoping = True

            # debug info
            cv2.putText(frame_bgr, f"dist: {distance:.2f} m",
                                (x - 40, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 1)
            
        # Nemame validni vzdalenost - tocime se na miste a hledame pylon
        # else:
            # print("Distance is None - searching for pylon")

        # print(f"linear={linear:.2f}, angular={angular:.2f}\n")
        # turtle.cmd_velocity(linear=linear, angular=angular)

        # --- VYKRESLENIE ---
        # Convert the 1-channel binary mask to a 3-channel BGR image to match the frame's shape for hstack
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame_bgr, mask_bgr))
        
        cv2.imshow("combined", combined)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


CIRCULARITY_THRESHOLD = 0.45



def find_pylon(frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray]:
    frame_bgr = frame.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([37, 70, 35])
    upper_green = np.array([80, 255, 255])

    # create a mask for green color and discard noise
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_coords = None

    if contours:
        # sort contours by area from largest to smallest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            
            # check circularity(roundness) of the contour
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity < CIRCULARITY_THRESHOLD:
                continue
            
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            x, y = int(x), int(y)

            # Save the coordinates of the largest contour (the first one processed)
            if target_coords is None:
                cv2.circle(frame_bgr, (x, y), 3, (0,0,255), -1)  
                target_coords = (x, y)

            # kreslenie - draw bounding circle and center for EVERY contour
            cv2.circle(frame_bgr, (x, y), int(radius), (0,255,0), 2)
            
            # Put text next to the object with area (A) and circularity (C)
            label = f"A:{area:.0f} C:{circularity:.2f}"
            cv2.putText(frame_bgr, label, (x + 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Return the target coordinates (if any were found), the drawn frame, and the mask
    return target_coords, frame_bgr, mask



if __name__ == '__main__':
    main()
