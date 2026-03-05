import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return hsv

def draw_hsv(hsv: np.ndarray) -> None:
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        print(f"pixel ({x},{y}) -> [{H[y,x]}, {S[y,x]}, {V[y,x]}]")

    plt.figure(figsize=(12,4))

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