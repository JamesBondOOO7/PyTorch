import numpy as np

def convolution(matrix, kernel):

    output_height = matrix.shape[0] - kernel.shape[0] + 1
    output_width = matrix.shape[1] - kernel.shape[1] + 1

    output_image = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            for ii in range(0, kernel.shape[0]):
                for jj in range(0, kernel.shape[1]):
                    output_image[i, j] += matrix[i + ii, j + jj]*kernel[ii, jj]

    return output_image

image = np.array([
    [0, 10, 10, 0],
    [20, 30, 30, 20],
    [10, 20, 20, 10],
    [0, 5, 5, 0],
])

filter = np.array([
    [1, 0],
    [0, 2]
])

output_image = convolution(image, filter)
print(output_image)

from scipy.signal import convolve2d
print(convolve2d(image, np.fliplr((np.flipud(filter))),mode='valid'))
