import cv2, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from solution import rgb_to_hsv

from sys import argv

hues = []

def onclick(event):

    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    print(f"pixel ({x},{y}) -> [{H[y,x]}, {S[y,x]}, {V[y,x]}]")
    hues.append((H[y,x], S[y,x], V[y,x]))


def main() -> None:
    global H, S, V

    # folder_path = os.path.join("sample_data", "recording_0001_mic_a_garaz_tma")
    folder_path = "training_data"
    print("Using folder:", folder_path)

    # i = int(argv[1])
    for item in os.listdir(folder_path):
        # filename = f"{i:04d}.mat"

        mat_file = os.path.join(folder_path, item) 
        data = scipy.io.loadmat(mat_file)

        rgb = data["image_rgb"]

        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # reconstruction
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]

        plt.figure(figsize=(24,8))

        plt.subplot(1,3,1)
        plt.imshow(H, cmap="hsv")
        plt.title("Hue")

        plt.subplot(1,3,2)
        plt.imshow(S, cmap="gray")
        plt.title("Saturation")

        plt.subplot(1,3,3)
        plt.imshow(V, cmap="gray")
        plt.title("Value")

        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()

    H_avg = sum([hue[0] for hue in hues]) / len(hues)
    S_avg = sum([hue[1] for hue in hues]) / len(hues)
    V_avg = sum([hue[2] for hue in hues]) / len(hues)

    print(f"Average values: [{H_avg}, {S_avg}, {V_avg}]")
    # ball : [73.7, 129.3, 109.85]


if __name__ == "__main__":
    main()