# Standard imports

# Third-party imports
import cv2
import numpy as np

"""
ToDo:
  Test all the functions
  Documentation
"""


class floodseg():
  """
  """
  def normalisation(self, cv2_img, ht_m):
    # Not sure if this is right: Normalizing image to max value of mean in the center area
    # Reduce the intensity of the image, kind of works though.. i leave it as it is for now
    return cv2.normalize(cv2_img, None, 0, ht_m,
                         norm_type=cv2.NORM_MINMAX)

  def spacecolourconversion(self, norimg):
    return cv2.cvtColor(norimg, cv2.COLOR_BGR2LAB)

  def binarisation(self, b_layer, ):
    h_t, otsu = cv2.threshold(b_layer, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (0.6 * h_t, h_t), cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

  def edgesdetection(self, img, t):
    # This threshold can be changed
    # This can be changed to "True/False" for increased/Decreased accuracy and speed
    return cv2.Canny(img, t[0], t[1], True)

  def add_x_images(self, images):
    no_images = len(images)
    if no_images < 2:
      return -1
    else:
      comb = cv2.add(images[0], images[1])
      if no_images >= 3:
        for img in images[2:]:
          comb = cv2.add(comb, img)
      return comb

  # TODO: This diff needs to be adjusted / set more dynamically
  def flooding_segmentation(self, img, H, W, diff=3):  # Taken from https://stackoverflow.com/a/46667829 :
    # The diff indicates: Maximal lower/upper brightness/color difference between the currently observed pixel and one of
    # its neighbors belonging to the component, or a seed pixel being added to the component
    # A lower diff works better ==> Can be adjusted...

    # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill

    seed = (W // 2, H // 2)

    mask = np.zeros((H + 2, W + 2), np.uint8)

    floodflags = 8  # No. Neigbours
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    # num, im, mask, rect = cv2.floodFill(img, mask, seed, (255, 0, 0), (diff,) * 3, (diff,) * 3, floodflags)
    num, im, mask, rect = cv2.floodFill(img, mask, seed, 255, diff, diff, floodflags)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((10, 10), np.uint8))  # Remove noise from the mask => Large kernel

  def get_ROI(self, mask):  # taken from https://stackoverflow.com/a/17507192
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the index of the largest contour: compact AF
    return cv2.boundingRect(contours[np.argmax([cv2.contourArea(c) for c in contours])])


def segment_image(img, offset=3, DEMO=False):
  """
    img: as numpy arrray
    offset: in percent (hand_target)
    DEMO: display the image with the mask => Default false
  """
  f = floodseg()

  H, W, _ = img.shape

  # STEP 1
  H_l = int((H // 2) * (1.00 - (offset * 0.01)))
  H_u = int((H // 2) * (1.00 + (offset * 0.01)))
  W_l = int((W // 2) * (1.00 - (offset * 0.01)))
  W_u = int((W // 2) * (1.00 + (offset * 0.01)))

  hand_target = img[H_l:H_u, W_l:W_u]
  ht_m = hand_target.mean(axis=(0, 1, 2))
  norimg = f.normalisation(img, ht_m)

  # STEP 2
  l, a, b = cv2.split(f.spacecolourconversion(norimg))

  # STEP 3
  t, b_bin = f.binarisation(a)  # Not sure whether it is best to use a or b here, can be changed
  # ... /Seems like a and b is swapped?
  # This works now.. dont change it plz

  # STEP 4
  l_edge = f.edgesdetection(img, t)  # supposed to use l layer here, but me thinks img works better

  # STEP 5
  sumimg = f.add_x_images([l, b_bin, l_edge])

  # STEP 6
  # TODO: Find the diff..
  mask = f.flooding_segmentation(sumimg, H, W)

  Xcrop, Ycrop, Wcrop, Hcrop = f.get_ROI(mask[2:H, 2:W])

  # DEMO functionality - SHOW THE MASK ON THE IMAGE:
  if DEMO:
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    imS = cv2.resize(cv2.rectangle(img, (Xcrop, Ycrop), (Xcrop + Wcrop, Ycrop + Hcrop), (0, 255, 0), 3),
                     (W // 4, H // 4))
    cv2.imshow("output", imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  foreground = cv2.bitwise_and(img, img, None, mask[0:H, 0:W])  # apply mask to image
  background = cv2.bitwise_and(img, img, None, cv2.bitwise_not(mask[0:H, 0:W]))  # invert the mask to get the background

  # returns croped image to the mask, foreground and background of image
  return cv2.bitwise_and(img, img, None, mask[0:H, 0:W])[Ycrop:Ycrop + Hcrop,
         Xcrop:Xcrop + Wcrop], foreground, background
