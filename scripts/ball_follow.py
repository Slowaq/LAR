from robolab_turtlebot import Turtlebot, Rate
import numpy as np
import cv2

WINDOW = "view"

def main():
    turtle = Turtlebot(rgb=True, pc=True)
    turtle.wait_for_rgb_image()

    response = 0.002
    rate = Rate(10)

    cv2.namedWindow("camera")
    cv2.namedWindow("depth")

    target_distance = 0.6  # 60 cm
    stoping = True


    while not turtle.is_shutting_down() and stoping == False:

        # --- RGB OBRAZ ---
        frame = turtle.get_rgb_image()
        if frame is None:
            continue

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- POINT CLOUD ---
        pc = turtle.get_point_cloud()
        image = np.zeros(pc.shape[:2])

        mask = np.logical_and(pc[:, :, 2] > 0.3, pc[:, :, 2] < 3.0)

        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)

        depth_vis = cv2.applyColorMap(
            255 - image.astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        if pc is None:
            continue

        linear = 0.0
        angular = 0.0
                
        error, x, y, frame = find_pylon(frame)
                
        if error is not None:
            angular = -response * error
        else:
            angular = 0.3  # hľadanie objektu

                # --- ZÍSKANIE VZDIALENOSTI Z POINT CLOUDU ---
                # ochrana proti indexu mimo rozsah
        h, w, _ = pc.shape
        if y >= h or x >= w:
            distance = None
        else:
            distance = pc[y, x, 2]  # Z = vzdialenosť dopredu

                # ak máme validnú vzdialenosť
        if distance is not None and not np.isnan(distance):

                    # riadenie dopredného pohybu
            if distance > target_distance:
                if abs(error) < 50:
                    linear = 0.1
                else:
                    linear = 0.0
            else:
                    linear = 0.0  # zastav
                    stoping = False

                    # debug info
            cv2.putText(frame, f"dist: {distance:.2f} m",
                                (x - 40, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 1)

        turtle.cmd_velocity(linear=linear, angular=angular)

        combined = np.hstack((frame, depth_vis))
        cv2.imshow("combined", combined)

        cv2.waitKey(1)

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

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

            return error, x, y, frame_bgr

    return None, None, None, frame_bgr
