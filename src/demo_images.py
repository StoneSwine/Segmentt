import os
import time

import cv2
# segmentation imports
from Segmentt.flooding import segment_image


def run_segment_image():
  # The directory containing this file
  HERE = os.path.dirname(__file__)

  times = []

  for filename in os.listdir(HERE + '/demo_imgs/'):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      oimg = cv2.imread(str(HERE + "/demo_imgs/" + filename), cv2.IMREAD_UNCHANGED)
      print(filename, oimg.shape)
      segment_image(oimg, 3, True)

  print("Running timing test")
  for filename in os.listdir(HERE + '/demo_imgs/'):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      oimg = cv2.imread(str(HERE + "/demo_imgs/" + filename), cv2.IMREAD_UNCHANGED)
      time1 = time.time()
      segment_image(oimg, 3, False)
      time2 = time.time()
      times.append((time2 - time1) * 1000.0)
      print('{:s} function took {:.2f} ms'.format(segment_image.__name__, times[-1]))

  print('Average time is {:.2f} ms'.format(sum(times) / len(times)))


if __name__ == "__main__":
  run_segment_image()
