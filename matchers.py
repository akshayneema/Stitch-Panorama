import cv2
import numpy as np 

class matchers:
	def __init__(self):
		self.orb = cv2.ORB_create(nfeatures = 1500, edgeThreshold = 15, patchSize = 15)
		self.ratio = 0.7
		self.reprojThresh = 4
		# old
		# self.surf = cv2.xfeatures2d.SURF_create()
		# FLANN_INDEX_KDTREE = 0
		# index_params = dict(algorithm=0, trees=5)
		# search_params = dict(checks=50)
		# self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def detectAndDescribe(self, image, name):
	    # descriptor = cv2.ORB_create(nfeatures = 1500, edgeThreshold = 15, patchSize = 15)
		kps = self.orb.detect(image, None)
		kps, features = self.orb.compute(image, kps)
		if (len(kps) > 0):
		    image = cv2.drawKeypoints(image,kps,np.array([]),color=(0,255,0), flags=0)
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
	    # compute the raw matches and initialize the list of actual
	    # matches
	    # print("in matchkeypoints")
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
		    # ensure the distance is within a certain ratio of each
		    # other (i.e. Lowe's ratio test)
		    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
		        matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
		    # construct the two sets of points
		    ptsA = np.float32([kpsA[i] for (_, i) in matches])
		    ptsB = np.float32([kpsB[i] for (i, _) in matches])

		    # compute the homography between the two sets of points
		    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
		        reprojThresh)
		    # print (H)
		    # return the matches along with the homograpy matrix
		    # and status of each matched point
		    # print len(matches)
		    return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None


	def match(self, imageA, imageB, direction=None):
		(kpsA, featuresA) = self.detectAndDescribe(imageA,"A")
		(kpsB, featuresB) = self.detectAndDescribe(imageB,"B")
		kpsA = np.float32([kpA.pt for kpA in kpsA])
		kpsB = np.float32([kpB.pt for kpB in kpsB])


		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, self.ratio, self.reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		return H
        

        # old
		# imageSet1 = self.getSURFFeatures(i1)
		# imageSet2 = self.getSURFFeatures(i2)
		# print "Direction : ", direction
		# matches = self.flann.knnMatch(
		# 	imageSet2['des'],
		# 	imageSet1['des'],
		# 	k=2
		# 	)
		# good = []
		# for i , (m, n) in enumerate(matches):
		# 	if m.distance < 0.7*n.distance:
		# 		good.append((m.trainIdx, m.queryIdx))

		# if len(good) > 4:
		# 	pointsCurrent = imageSet2['kp']
		# 	pointsPrevious = imageSet1['kp']

		# 	matchedPointsCurrent = np.float32(
		# 		[pointsCurrent[i].pt for (__, i) in good]
		# 	)
		# 	matchedPointsPrev = np.float32(
		# 		[pointsPrevious[i].pt for (i, __) in good]
		# 		)

		# 	H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
		# 	return H
		# return None

	# def getSURFFeatures(self, im):
	# 	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# 	kp, des = self.surf.detectAndCompute(gray, None)
	# 	return {'kp':kp, 'des':des}