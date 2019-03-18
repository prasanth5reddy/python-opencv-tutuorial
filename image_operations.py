import cv2
import numpy as np
import matplotlib.pyplot as plt

# read an image
img = cv2.imread('data/lenna.png', 1)

# display and image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# shape and size
rows, columns, channels = img.shape
print(rows, columns, channels)
print(img.size)
print(rows * columns * channels == img.size)

# class
print(img.dtype)

# accessing pixel values
px = img[100, 100]
print(px)
blue = img[100, 100, 0]  # follows BGR
print(blue)

# accessing entire row
img = cv2.imread('data/lenna.png', 0)
row = img[100, :]
plt.plot(row)
plt.show()

# cropping an image
cropped_img = img[20:200, 40:160]
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(cropped_img.shape)

# adding two images
lenna = img
mandrill = cv2.imread('data/mandrill.png', 0)
mandrill = mandrill[:220, :220]
print(lenna.shape, mandrill.shape)
summed = lenna + mandrill
cv2.imshow('Summed Image', summed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# average of images
average = (lenna / 2).astype(np.uint8) + (mandrill / 2).astype(np.uint8)
cv2.imshow('Average Image', average)
cv2.waitKey(0)
cv2.destroyAllWindows()

average_alt = ((lenna + mandrill) / 2).astype(np.uint8)  # note that pixel value can hold a maximum value of 255
cv2.imshow('Average Alt Image', average_alt)
cv2.waitKey(0)
cv2.destroyAllWindows()

# multiplying by a scalar
scaled_img = (img * 1.5).astype(np.uint8)
cv2.imshow('Scaled Image', scaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blend images
blended_img = (lenna * 0.75).astype(np.uint8) + (mandrill * 0.25).astype(np.uint8)
cv2.imshow('Blended Image', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
