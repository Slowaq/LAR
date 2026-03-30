import sys
import os
import math
import time
import signal
import matplotlib.pyplot as plt
import cv2
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.solution.algorithm import Algorithm

# ===== GLOBAL (for safe shutdown) =====
algo = None

def signal_handler(sig, frame):
    global algo
    print("\nCtrl+C detected, stopping robot and exiting...")

    if algo is not None:
        try:
            algo.robot.cmd_velocity(0, 0)
        except:
            pass

    cv2.destroyAllWindows()
    sys.exit(0)

# register Ctrl+C handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    global algo
    algo = Algorithm()

    print("Starting fixed-rectangle odometry test... (Ctrl+C to stop)")

    algo.robot.reset_odometry()
    algo.robot.wait_for_odometry()
    cv2.waitKey(200)

    start = algo.robot.get_odometry()
    if start is None:
        print("Odometry is None")
        return

    sx, sy, sphi = start

    # ===== RECTANGLE DEFINITION (GLOBAL FRAME) =====
    width = 1
    height = 1

    rectangle_points = [
        (sx + width, sy),
        (sx + width, sy + height),
        (sx, sy + height),
        (sx, sy),
    ]

    N = 15  # number of repetitions

    all_paths = []

    for i in range(N):
        print(f"\n=== Rectangle run {i + 1}/{N} ===")

        current = algo.robot.get_odometry()
        if current is None:
            print("Odometry is None")
            break

        algo.trajectory = [(current[0], current[1])]

        for (px, py) in rectangle_points:
            print(f"Going to ({px:.2f}, {py:.2f})")

            ok = algo._go_to_point_using_odometry(px, py)
            if not ok:
                print("Movement failed")
                break

            time.sleep(0.05)  # helps Ctrl+C responsiveness

        path = algo.trajectory.copy()
        all_paths.append(path)

    # ===== SAVE TRAJECTORIES TO FILE =====
    save_file = "trajectories.json"
    with open(save_file, "w") as f:
        json.dump(all_paths, f)
    print(f"All trajectories saved to {save_file}")

    # ===== STOP ROBOT AFTER FINISH =====
    try:
        algo.robot.cmd_velocity(0, 0)
    except:
        pass

    # ===== PLOTTING =====
    plt.figure()

    # plot all trajectories with a single legend entry
    for path in all_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color='b', label="Trajectory")
    # prevent duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # reference rectangle (dashed)
    rx = [p[0] for p in rectangle_points] + [rectangle_points[0][0]]
    ry = [p[1] for p in rectangle_points] + [rectangle_points[0][1]]
    plt.plot(rx, ry, 'k--', label="Reference Square")
    plt.legend(by_label.values(), by_label.keys())

    # start point
    plt.scatter([sx], [sy], c='r', label="Start")

    plt.axis("equal")
    plt.grid(True)

    # non-blocking show (so Ctrl+C still works)
    plt.show()  # blocking=True by default
    input("Press ENTER to exit...")
    plt.close()


if __name__ == "__main__":
    main()