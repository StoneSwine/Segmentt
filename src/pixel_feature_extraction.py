#!/usr/bin/env python3

import math
# Standard library imports
import os

# Third party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# finseg imports
from Segmentt.flooding import segment_image

WSIZ=50

def local_coherence(img, window_s=WSIZ):
  """
  Calculate the coherence according to methdology described in:
    Bazen, Asker M., and Sabih H. Gerez. "Segmentation of fingerprint images."
    ProRISC 2001 Workshop on Circuits, Systems and Signal Processing. Veldhoven,
    The Netherlands, 2001.
  """
  coherence = []
  rs = window_s
  cs = window_s
  for r in range(4, img.shape[0] - rs, rs):
    for c in range(4, img.shape[1] - cs, cs):
      window = img[r:r + rs, c:c + cs]
      if window.var() != 0: # Need variance because of the constraint (gxx + gyy) < 0
        gx = np.uint8(np.absolute(cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=5))).flatten()
        gy = np.uint8(np.absolute(cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=5))).flatten()

        gxx = sum([int(x) ** 2 for x in gx])
        gyy = sum([int(y) ** 2 for y in gy])
        gxy = sum([int(x) * int(y) for x, y in zip(gx, gy)])

        assert gxx + gyy != 0
        coherence.append(math.sqrt((math.pow((gxx - gyy), 2) + 4 * math.pow(gxy, 2))) / (gxx + gyy))
  return coherence


def local_variance_mean(img, window_s=WSIZ):
  """
  Calculate local variance and mean values for a window using the numpy variance and mean functions
  """
  rs = cs = window_s
  mean = []
  variance = []
  for r in range(4, img.shape[0] - rs, rs):
    for c in range(4, img.shape[1] - cs, cs):
      window = img[r:r + rs, c:c + cs]
      if window.var() != 0:
        mean.append(np.mean(window))
        variance.append(np.var(window))
  return variance, mean


def run_segment_image():
  """
  This program is kind of slow because of all the looping needed to go through a full image
  """
  HERE = os.path.dirname(__file__)

  fg_v = []
  fg_m = []
  fg_c = []
  bg_v = []
  bg_m = []
  bg_c = []

  for filename in os.listdir(HERE + '/baseline/'):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      oimg = cv2.imread(HERE + '/baseline/' + filename, cv2.IMREAD_UNCHANGED)
      croppedforeground, _, background = segment_image(oimg)
      print(filename)

      variance, mean = local_variance_mean(croppedforeground)
      for x in variance:
        fg_v.append(x)
      for x in mean:
        fg_m.append(x)
      for x in local_coherence(croppedforeground):
       fg_c.append(x)

      variance, mean = local_variance_mean(background)
      for x in variance:
        bg_v.append(x)
      for x in mean:
        bg_m.append(x)
      for x in local_coherence(background):
        bg_c.append(x)

  plt.title("Variance")
  sns.distplot(fg_v, hist=False, label="Foreground")
  sns.distplot(bg_v, hist=False, label="Background")
  plt.show()

  plt.title("Mean")
  sns.distplot(fg_m, hist=False, label="Foreground")
  sns.distplot(bg_m, hist=False, label="Background")
  plt.show()

  plt.title("Coherence")
  sns.distplot(fg_c, hist=False, label="Foreground")
  sns.distplot(bg_c, hist=False, label="Background")
  plt.show()


if __name__ == "__main__":
  run_segment_image()
