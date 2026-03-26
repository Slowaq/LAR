from robolab_turtlebot import Turtlebot, Rate
import numpy as np
import cv2

WINDOW = "view"

def main():
    print("main started")
    turtle = Turtlebot(rgb=True, pc=True)
    turtle.wait_for_rgb_image()

    RESPONSE = 0.003
    rate = Rate(10)

    cv2.namedWindow("camera")
    cv2.namedWindow("depth")

    print("window created")

    target_distance = 0.8  # 80 cm
    stoping = False


    while not turtle.is_shutting_down() and not stoping:
        # --- RGB OBRAZ ---
        turtle.wait_for_rgb_image()
        frame = turtle.get_rgb_image()
        if frame is None:
            print("No RGB image - this should not happen")
            continue

        # --- POINT CLOUD ---
        turtle.wait_for_point_cloud()
        pc = turtle.get_point_cloud()
        if pc is None:
            print("Pointcloud is None - this should never happen")
            continue

        image = np.zeros(pc.shape[:2])

        mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

        depth_vis = cv2.applyColorMap(
            255 - image.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        linear = 0.0
        angular = 0.0
                
        error, x, y, frame = find_pylon(frame)
                
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
        if distance is not None:

            # riadenie dopredného pohybu    
            print(f"distance: {distance:.2f}, target_distace: {target_distance:.2f}")
            if distance > target_distance:
                if abs(error) < 50:
                    print("Going after target")
                    linear = 0.1    # jedem dopredu
                else:
                    print("Angular error is too large - turning on the spot")
                    linear = 0.0    # jenom otaceni
            else:
                    print("Close enough to the target - stopping")
                    linear = 0.0  # zastav
                    stoping = True

                    # debug info
            cv2.putText(frame, f"dist: {distance:.2f} m",
                                (x - 40, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 1)
            
        # Nemame validni vzdalenost - tocime se na miste a hledame pylon
        else:
            print("Distance is None - searching for pylon")

        turtle.cmd_velocity(linear=linear, angular=angular)

        combined = np.hstack((frame, depth_vis))
        cv2.imshow("combined", combined)

        cv2.waitKey(1)

        rate.sleep()

    cv2.destroyAllWindows()



def find_pylon(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 70, 70])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 300:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            x, y = int(x), int(y)

            width = frame.shape[1]
            center_x = width // 2
            error = x - center_x

            # kreslenie
            cv2.circle(frame_bgr, (x, y), int(radius), (0,255,0), 2)
            cv2.circle(frame_bgr, (x, y), 3, (0,0,255), -1)
            cv2.line(frame_bgr, (center_x, 0),
                     (center_x, frame.shape[0]), (255,0,0), 2)

            print(f"Found pylon: error={error:.2f}, x={x:.2f}, y={y:.2f}")
            return error, x, y, frame_bgr

    print("Couldnt find pylon")
    return None, None, None, frame_bgr



if __name__ == '__main__':
    main()
