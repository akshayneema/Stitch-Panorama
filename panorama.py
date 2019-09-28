# import the necessary packages
import numpy as np
import imutils
import cv2
count=0
class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
    def stitchleft(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        # print("in stitch")
        global count
        count=count+1
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA,"A")
        (kpsB, featuresB) = self.detectAndDescribe(imageB,"B")
        kpsA = np.float32([kpA.pt for kpA in kpsA])
        kpsB = np.float32([kpB.pt for kpB in kpsB])


        # match features between the two images
        M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        print H
        print imageA.shape
        C=np.dot(H,np.array([0,0,1]))
        C=C/C[2]
        
        A=np.dot(H,np.array([imageB.shape[1],0,1]))
        A=A/A[2]
        D=np.dot(H,np.array([0,imageB.shape[0],1]))
        D=D/D[2]
        B=np.dot(H,np.array([imageB.shape[1],imageA.shape[0],1]))
        B=B/B[2]
        print C
        # print imageA.shape[1]
        print A
        print D
        print B
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # result = cv2.warpPerspective(imageA, H, (int(max(A[0],B[0])), imageA.shape[0]))
        cv2.imshow("Result1"+str(count), result)
        # scale=float(((A[0]+B[0])/2)-((C[0]+D[0])/2))/float(imageA.shape[1])
        # print "numerator ",(((A[0]+B[0])/2)-((C[0]+D[0])/2))
        # print "denominator ",imageA.shape[1]
        # print "scale ",scale
        # result[:,imageB.shape[1]:imageB.shape[1]+int((max(A[0],B[0])-imageB.shape[1])/scale)] = cv2.resize(result[:,imageB.shape[1]:int(max(A[0],B[0]))], (int((max(A[0],B[0])-imageB.shape[1])/scale),result.shape[0]), interpolation = cv2.INTER_AREA)
        # cv2.imshow("Result2"+str(count), result)
        # result=result[:,:imageB.shape[1]+int((max(A[0],B[0])-imageB.shape[1])/scale)]
        # cv2.imshow("Result3"+str(count), result)
        # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # cv2.imshow("Result4"+str(count), result)
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        # print("in stitch")
        global count
        count=count+1
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA,"A")
        (kpsB, featuresB) = self.detectAndDescribe(imageB,"B")
        kpsA = np.float32([kpA.pt for kpA in kpsA])
        kpsB = np.float32([kpB.pt for kpB in kpsB])


        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        print H
        print imageA.shape
        C=np.dot(H,np.array([0,0,1]))
        C=C/C[2]
        
        A=np.dot(H,np.array([imageA.shape[1],0,1]))
        A=A/A[2]
        D=np.dot(H,np.array([0,imageA.shape[0],1]))
        D=D/D[2]
        B=np.dot(H,np.array([imageA.shape[1],imageA.shape[0],1]))
        B=B/B[2]
        print C
        # print imageA.shape[1]
        print A
        print D
        print B
        # result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result = cv2.warpPerspective(imageA, H, (int(max(A[0],B[0])), imageA.shape[0]))
        # cv2.imshow("Result1"+str(count), result)
        scale=float(((A[0]+B[0])/2)-((C[0]+D[0])/2))/float(imageA.shape[1])
        print "numerator ",(((A[0]+B[0])/2)-((C[0]+D[0])/2))
        print "denominator ",imageA.shape[1]
        print "scale ",scale
        result[:,imageB.shape[1]:imageB.shape[1]+int((max(A[0],B[0])-imageB.shape[1])/scale)] = cv2.resize(result[:,imageB.shape[1]:int(max(A[0],B[0]))], (int((max(A[0],B[0])-imageB.shape[1])/scale),result.shape[0]), interpolation = cv2.INTER_AREA)
        # cv2.imshow("Result2"+str(count), result)
        result=result[:,:imageB.shape[1]+int((max(A[0],B[0])-imageB.shape[1])/scale)]
        # cv2.imshow("Result3"+str(count), result)
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # cv2.imshow("Result4"+str(count), result)
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image, name):
        # print("in detect and describe")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # descriptor = cv2.xfeatures2d.SIFT_create()
        descriptor = cv2.ORB_create(nfeatures = 1500, edgeThreshold = 15, patchSize = 15)
        kps = descriptor.detect(image, None)
        kps, features = descriptor.compute(image, kps)
        # print (len(kps))
        # print (descriptor.getMaxFeatures())
        # kps = np.float32([kp.pt for kp in kps])

        # cv2.imshow("kps:",kps)
        # cv2.imshow("features",features)
        if (len(kps) > 0):
            image = cv2.drawKeypoints(image,kps,np.array([]),color=(0,255,0), flags=0)
        # cv2.imshow("keypoints" + name,image)
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

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
if __name__== "__main__":
    imageA = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(1)+'.jpg')
    imageB = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(6)+'.jpg')
    imageC = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(8)+'.jpg')
    imageD = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(2)+'.jpg')
    imageE = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(3)+'.jpg')
    imageF = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(4)+'.jpg')
    imageG = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(5)+'.jpg')
    imageH = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(7)+'.jpg')
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)
    imageC = imutils.resize(imageC, width=400)
    imageD = imutils.resize(imageD, width=400)
    imageE = imutils.resize(imageE, width=400)
    imageF = imutils.resize(imageF, width=400)
    imageG = imutils.resize(imageG, width=400)
    imageH = imutils.resize(imageH, width=400)
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Image C", imageC)
    # cv2.imshow("Image D", imageD)
    cv2.imshow("Image E", imageE)
    cv2.imshow("Image F", imageF)
    cv2.imshow("Image G", imageG)
    cv2.imshow("Image H", imageH)
    # print(str(i)+" "+str(j))
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    ans = stitcher.stitch([imageG, imageH], showMatches=True)
    (result, vis) = ans
    # cv2.imshow("Keypoint Matches1", vis)
    cv2.imshow("Result1", result)
    ans = stitcher.stitch([imageF, result], showMatches=True)
    (result, vis) = ans
    cv2.imshow("Result2", result)
    ans = stitcher.stitch([imageE, result], showMatches=True)
    (result, vis) = ans
    cv2.imshow("Result3", result)
    # ans = stitcher.stitch([imageD, result], showMatches=True)
    # (result, vis) = ans
    # cv2.imshow("Result4", result)
    # ans = stitcher.stitch([imageC, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageB, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageA, result], showMatches=True)
    # (result, vis) = ans
    # cv2.imshow("Keypoint Matches2", vis)
    # (kps, features) = stitcher.detectAndDescribe(imageA,"A")
    # (kps, features) = stitcher.detectAndDescribe(imageB,"B")

    # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Keypoint Matches", vis)
    # cv2.imshow("Result2", result)
    cv2.waitKey(0)