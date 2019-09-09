import numpy as np
import cv2 as cv
import imutils

# import matplotlib.pyplot as plt
img1 = cv.imread('./Data/1/2.jpg')          # queryImage
img2 = cv.imread('./Data/1/3.jpg') # trainImage
img1 = imutils.resize(img1, width=400)
img2 = imutils.resize(img2, width=400)
cv.imshow("image A",img1)
cv.imshow("image B",img2)

# Initiate ORB detector
orb = cv.ORB_create(nfeatures = 1500, edgeThreshold = 15, patchSize = 15)
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print ("number of keypoint in image1: ", len(kp1))
print ("number of keypoint in image2: ", len(kp2))

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#matches on the images
cv.imshow("keypoints matches",img3)
#finding the homography
queryIdx_list = [m.queryIdx for m in matches]
trainIdx_list = [m.trainIdx for m in matches]
imgIdx_list = [m.imgIdx for m in matches]
distance_list = [m.distance for m in matches]
ptsA = np.float32([kp1[i].pt for i in queryIdx_list])
ptsB = np.float32([kp2[i].pt for i in trainIdx_list])
(H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, 4.0)
print (H)
# move the image2 over image1
height_img1,width_img1,channels_img1 = img1.shape
# im1_out = cv.warpPerspective(img1, H, (width,height))
# cv.imshow("final image1",im1_out)
height_img2,width_img2,channels_img2 = img2.shape
im_out = cv.warpPerspective(img2, H, (width_img1 + width_img2, height_img1 + height_img2))
cv.imshow("final image",im_out)
cv.waitKey(0)
