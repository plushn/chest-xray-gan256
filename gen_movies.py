import glob
import cv2
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

img_array = []
for filename in sorted(glob.glob("generated_images/*00.png"), key=natural_keys):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

name = 'animation.mp4'
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()