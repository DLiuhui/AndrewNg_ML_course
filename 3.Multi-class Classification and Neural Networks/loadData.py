# using scipy.io to read .mat file
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# data size (5000, 400)
# label (400, 1)
def loadData(file):
    data = scipy.io.loadmat(file)
    x = data['X']
    labels = data['y']
    return x, labels

# theta1 (25, 401)
# theta2 (10, 26)
def loadWeight(file):
    weight = scipy.io.loadmat(file)
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']
    return theta1, theta2

def displayData(data):
    (numbers, pixels) = data.shape
    # the shape of sample pic is 20*20
    # get pic height, width
    pic_width = np.round(np.sqrt(pixels)).astype(int)
    pic_height = (pixels / pic_width).astype(int)
    # compute the number of items to display
    display_rows = np.floor(np.sqrt(numbers)).astype(int)
    display_cols = np.ceil(numbers / display_rows).astype(int)
    # between images padding
    pad = 1
    # display panel
    display_array = - np.ones((pad + display_rows * (pic_height + pad),
                               pad + display_cols * (pic_width + pad)))
    # copy each example into a patch on the display array
    count = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if count == numbers:
                break
            max_val = np.max(np.abs(data[count]))
            display_array[(pad + j * (pic_height + pad)) : (pad + j * (pic_height + pad) + pic_height),
            (pad + i * (pic_width + pad)) : (pad + i * (pic_width + pad) + pic_width)] \
                = data[count].reshape((pic_height, pic_width)) / max_val
            count += 1
        if count == numbers:
            break

    # display img & save
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
    plt.savefig('data_show.png')