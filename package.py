import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import sys


def imfill (im_in):
  #Dynamic Threshholding with mean
  mean = np.mean(im_in)
  #print mean

  th, im_th = cv2.threshold(im_in, mean, 255, cv2.THRESH_BINARY_INV);

  # Copy the thresholded image.
  im_floodfill = im_th.copy()
   
  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = im_th.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
   
  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);
   
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
   
  # Combine the two images to get the foreground.
  im_out = im_th | im_floodfill_inv
  return im_out


def imclearborder(imggray, radius):
    imggraycopy = imggray.copy()
    
    #Dynamic Threshholding with mean
    mean = np.mean(imggray)
    #print mean
    
    ret, thresh = cv2.threshold(imggray, mean, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imggray.shape[0]
    imgCols = imggray.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imggraycopy, contours, idx, (0,0,0), -1)

    return imggraycopy


def main(arg):
  precision=1       #hough precision
  img = cv2.imread(arg)
  # OpenCV uses BGR, matplotlib uses RGB - so must convert  
  RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(RGB_img)
  plt.show()
  cv2.imwrite('original.jpg',RGB_img)

  rows, cols, channels = img.shape
  #print height, width, channels
  
  if (channels > 1):
    b,g,r = cv2.split(img)  #get RGB values 

  #Convert to gray
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  plt.imshow(gray, cmap='gray')
  plt.show()
  cv2.imwrite('gray.jpg',gray)

#PreProcessing
  #clear Border
  bw = imclearborder(gray,4);  #get rid of the border ************* 1
  plt.imshow(bw, cmap='gray')
  plt.show()
  cv2.imwrite('clearborder.jpg',bw)

  #fill holes
  bw = imfill(bw);      #fill holes ************** 2
  plt.imshow(bw, cmap='gray')
  plt.show()
  cv2.imwrite('fillholes.jpg',bw)

  #Apply Canny
  #Need to Adjust the size of the image to prevent the edges of the picture from being extracted
  BW = cv2.Canny(bw, 100, 200, apertureSize=3) #detect edges ************ 3 -- can use canny
  plt.imshow(BW, cmap='gray')
  plt.show()
  cv2.imwrite('canny.jpg',BW)

  #Apply Hough
  lines = cv2.HoughLines(BW,1,np.pi/180,200)
  print lines
  for rho,theta in lines[0]:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 100*(-b))
      y1 = int(y0 + 100*(a))
      x2 = int(x0 - 100*(-b))
      y2 = int(y0 - 100*(a))
      print [x0,y0],[x1,y1],[x2,y2]
      cv2.line(BW,(x1,y1),(x2,y2),(0,0,255),2)

  #TODO
  #How to Adjust Hough Point ???? for Rotation ???  
  pts1 = np.float32([[50,50],[200,50],[50,200]])
  pts2 = np.float32([[10,100],[200,50],[100,250]])
  # pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2]])
  # pts2 = np.float32([[x0,y0],[x1,y1],[x2,y2]])
  Ta1 = cv2.getAffineTransform(pts1,pts2)
  Ia1 = cv2.warpAffine(img,Ta1,(cols,rows))

  plt.imshow(Ia1, cmap='gray') 
  plt.show()
  cv2.imwrite('rotate.jpg',Ia1)

  #after rotating
  rows, cols, channels = Ia1.shape
  if (channels > 1):
    b,g,r = cv2.split(Ia1)  #get RGB values 
  gray1 = cv2.cvtColor(Ia1, cv2.COLOR_BGR2GRAY)
  bw1 = imclearborder(gray1,2);  #get rid of the border ************* 1
  plt.imshow(bw1, cmap='gray') 
  plt.show()
  cv2.imwrite('process_roated.jpg',bw1)

#  bw1 = imfill(bw1);      #fill holes ************** 2
  # plt.imshow(bw1, cmap='gray') 
  # plt.show()

  #Cropping Needs to be done

  cv2.imwrite('hough.jpg',BW)

if __name__ == "__main__":
  try:
    arg = sys.argv[1]
  except:
    print 'needs one argument parameter for image path'
  main(arg)
