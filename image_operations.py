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

# adding images using OpenCV function
summed_cv = cv2.add(lenna, mandrill)
cv2.imshow('Summed Image CV', summed_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(lenna[0, 0], mandrill[0, 0], summed[0, 0], summed_cv[0, 0])

# The above two summed images are different so use cv functions otherwise we have to convert type every time

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

# blend using OpenCV function
blended_cv = cv2.addWeighted(lenna, 0.75, mandrill, 0.25, 0)
cv2.imshow('Blended Image CV', blended_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# adding noise to image
noised_img = lenna + (np.random.normal(size=[lenna.shape[0], lenna.shape[1]]) * 10).astype(np.uint8)
cv2.imshow('Noised Image', noised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

noised_img_cv = cv2.add(lenna, (np.random.normal(size=[lenna.shape[0], lenna.shape[1]]) * 0.5).astype(np.uint8))
cv2.imshow('Noised Image CV', noised_img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# subtraction of images
diff_img = mandrill - lenna
cv2.imshow('Difference Image', diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

abs_diff_img = abs(mandrill - lenna)
# here subtraction is performed first hence this and above both are same. See below to handle this
cv2.imshow('Abs Difference Image', abs_diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# just the difference adjusted to uint8 type
diff_img_cv = cv2.subtract(mandrill, lenna)
cv2.imshow('Difference Image CV', diff_img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(mandrill[0, 0], lenna[0, 0], diff_img[0, 0], abs_diff_img[0, 0], diff_img_cv[0, 0])

# True difference of two images
diff_img_true_cv = cv2.subtract(mandrill, lenna) + cv2.subtract(lenna, mandrill)
cv2.imshow('Difference Image True CV', diff_img_true_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
