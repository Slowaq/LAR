import cv2, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def onclick(event):

    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    print(f"pixel ({x},{y}) -> H={H[y,x]}, S={S[y,x]}, V={V[y,x]}")


def main() -> None:
    global H, S, V
    folders = sorted(os.listdir("sample_data"))
    latest_folder = folders[-1]

    folder_path = os.path.join("sample_data", latest_folder)
    print("Using folder:", folder_path)

    files = [f for f in os.listdir(folder_path)]
    for f in files:
        mat_file = os.path.join(folder_path, files[0])
        data = scipy.io.loadmat(mat_file)

        rgb = data["image_rgb"]

        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        print(len(hsv), len(hsv[0]))


    
    # reconstruction
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

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

if __name__ == "__main__":
    main()