import time
import cv2 as cv
import numpy as np


print('Starting background estimation')
vc = cv.VideoCapture('Data/videoplayback.mp4')
length = int(vc.get(cv.CAP_PROP_FRAME_COUNT))

images = np.array([vc.read()[1] for i in range(length)])
selected_images = images[380:2090]
del images 
images = selected_images


image_size = images[0].shape
memory_size = np.prod(images.shape)
print('Memory size: {:.2f} MB ({} images)'.format(memory_size / 1024**2, length))

start_time = time.time()
background = np.median(images, axis=0)
end_time = time.time()

print('Took {:.2f} seconds to compute background'.format(end_time - start_time))

for img in images:
    diff = np.subtract(img, background)
    diff = np.abs(diff).astype(np.uint8)
    cv.imshow('diff', diff)
    cv.waitKey(10)
