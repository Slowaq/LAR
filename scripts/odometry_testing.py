import sys
import os
import math
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.solution.algorithm import Algorithm


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def wait_for_b1(robot):
    print("Press B1 to start the test")
    while not robot.is_shutting_down():
        if robot.get_button("B1"):  
            while robot.get_button("B1"): 
                pass
            return

def main():
    algo = Algorithm()

    wait_for_b1(algo.robot)

    algo.robot.reset_odometry()
    algo.robot.wait_for_odometry()
    cv2.waitKey(200)

    start = algo.robot.get_odometry()
    if start is None:
        print("Odometry is None")
        return

    sx, sy, _ = start
    target = (sx + 0.5, sy + 0.5)   # 1 m forward from start

    all_paths = []
    final_points = []
    errors = []

    N = 5  # number of repeated trials

    for i in range(N):
        print(f"\nTrial {i + 1}/{N}")

        algo.trajectory = [(sx, sy)]

        ok = algo._go_to_point_using_odometry(target[0], target[1])
        current = algo.robot.get_odometry()
        if current is None:
            print("Odometry is None after motion")
            break

        path = algo.trajectory.copy()
        all_paths.append(path)

        end_point = (current[0], current[1])
        final_points.append(end_point)
        error = dist(end_point, target)
        errors.append(error)

        print(f"Reached: {ok}, end error = {error:.3f} m")

    plt.figure()
    for path in all_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, marker="r-")

    plt.scatter([sx], [sy], label="start")
    plt.scatter([target[0]], [target[1]], label="target")
    if final_points:
        fx = [p[0] for p in final_points]
        fy = [p[1] for p in final_points]
        plt.scatter(fx, fy, label="final points")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    if errors:
        print(f"\nMean final error: {sum(errors) / len(errors):.3f} m")
        print(f"Max final error:  {max(errors):.3f} m")


if __name__ == "__main__":
    main()