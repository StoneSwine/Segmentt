#!/usr/bin/env python3

import os

import cv2
import matplotlib.pyplot as plt

# finseg imports
from Segmentt.flooding import segment_image


def run_segment_image():
  # The directory containing this file
  HERE = os.path.dirname(__file__)
  bh_maxv = []
  bw_maxv = []

  # The baseline images must be inspected manually to make sure that they are segmented correctly
  print("#Baseline:")
  for filename in os.listdir(HERE + '/baseline/'):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      oimg = cv2.imread(HERE + '/baseline/' + filename, cv2.IMREAD_UNCHANGED)
      croppedimage, _, _ = segment_image(oimg)

      print(filename, croppedimage.shape)

      bh_maxv.append(croppedimage.shape[0])
      bw_maxv.append(croppedimage.shape[1])

  th_maxv = []
  tw_maxv = []
  correct = 0

  # Testinng images and comparing to baseline
  print("\n#Test")
  for filename in os.listdir(HERE + '/demo_imgs/'):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      oimg = cv2.imread(HERE + '/demo_imgs/' + filename, cv2.IMREAD_UNCHANGED)
      croppedimage, _, _ = segment_image(oimg)

      print(filename, croppedimage.shape)

      th_maxv.append(croppedimage.shape[0])
      tw_maxv.append(croppedimage.shape[1])

      if min(bh_maxv) <= croppedimage.shape[0] <= max(bh_maxv) and min(bw_maxv) <= croppedimage.shape[1] <= max(
              bw_maxv):
        correct += 1

  print("\nPercent correct:         {:.2f} %".format((correct / len(th_maxv)) * 100))
  print("No. correctly segmented: {}".format(correct))

  plt.title("Size distributions")
  plt.scatter(th_maxv, tw_maxv, label="Testing image", linewidths=1, marker='x')
  plt.plot([max(bh_maxv), max(bh_maxv), min(bh_maxv), min(bh_maxv), max(bh_maxv)],
           [max(bw_maxv), min(bw_maxv), min(bw_maxv), max(bw_maxv), max(bw_maxv)], color='orange', marker='o',
           linestyle='dashed', label="Boundary b")
  plt.legend(loc="upper left")
  plt.gca().set_xbound(lower=500)  # Must be adjusted depending on the size of the image:
  plt.gca().set_ybound(lower=1500)  # Must be adjusted depending on the size of the image:
  plt.xlabel("Height")
  plt.ylabel("Width")
  plt.show()


if __name__ == "__main__":
  run_segment_image()
