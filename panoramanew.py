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
        global count
        count=count+1
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        # print("in stitch")
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            # print("return None")
            return None
        # print("here")
        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        # Hn = np.linalg.inv(H)
        
        # Hn = H
        print H
        print imageB.shape
        C=np.dot(H,np.array([0,0,1]))
        C=C/C[2]
        
        A=np.dot(H,np.array([imageB.shape[1],0,1]))
        A=A/A[2]
        D=np.dot(H,np.array([0,imageB.shape[0],1]))
        D=D/D[2]
        B=np.dot(H,np.array([imageB.shape[1],imageB.shape[0],1]))
        B=B/B[2]
        print C
        # print imageA.shape[1]
        print A
        print D
        print B
        # H[0][2]=H[0][2]-min(C[0],D[0])
        imageA = cv2.copyMakeBorder(imageA,top=0,bottom=0,left=int(-min(C[0],D[0])),right=0,borderType=cv2.BORDER_CONSTANT,value=[0, 0, 0])
        # cv2.imshow("changed image", imageA)
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)
        (matches, H, status) = M
        result = (cv2.warpPerspective(imageB, H, (imageA.shape[1], imageB.shape[0])))
        # cv2.imshow("Result1"+str(count), result)
        scale=float(max(A[0],B[0])-min(C[0],D[0]))/float(imageB.shape[1])
        print "numerator ",(max(A[0],B[0])-min(C[0],D[0]))
        print "denominator ",imageA.shape[1]
        print "scale ",scale
        result[:,int(-min(C[0],D[0]))-int(-min(C[0],D[0])/scale):int(-min(C[0],D[0]))] = cv2.resize(result[:,0:int(-min(C[0],D[0]))], (int(-min(C[0],D[0])/scale),result.shape[0]), interpolation = cv2.INTER_AREA)
        # cv2.imshow("Result2"+str(count), result)
        result=result[:,int(-min(C[0],D[0]))-int(-min(C[0],D[0])/scale):]
        # cv2.imshow("Result3"+str(count), result)
        result[:, int(-min(C[0],D[0])/scale):] = imageA[:,int(-min(C[0],D[0])):]
        # cv2.imshow("Result4"+str(count), result)

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        global count
        count=count+1
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        # print("in stitch")
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            # print("return None")
            return None
        # print("here")
        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        # Hn = np.linalg.inv(H)
        Hn = H
        print Hn
        print imageA.shape
        C=np.dot(Hn,np.array([0,0,1]))
        C=C/C[2]
        
        A=np.dot(Hn,np.array([imageA.shape[1],0,1]))
        A=A/A[2]
        D=np.dot(Hn,np.array([0,imageA.shape[0],1]))
        D=D/D[2]
        B=np.dot(Hn,np.array([imageA.shape[1],imageA.shape[0],1]))
        B=B/B[2]
        print C
        # print imageA.shape[1]
        print A
        print D
        print B
        result = cv2.warpPerspective(imageA, H, (int(max(A[0],B[0])), imageA.shape[0]))
        # result = cv2.warpPerspective(imageA, H, (imageB.shape[1]+imageA.shape[1], imageA.shape[0]))
        # cv2.imshow("Result1"+str(count), result)
        # if(count==2):
        #     vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            
        #     return (result,vis)
        # height=imageA.shape[0]
        # width=imageA.shape[1] + imageB.shape[1]
        # awidth=0
        # for i in range(0,height):
        #     for j in range(width-1,-1,-1):
        #         if result[i][j][0]!=0 or result[i][j][1]!=0 or result[i][j][2]!=0:
        #             # minwidth=min(minwidth,j)
        #             awidth+=j
        #             break
        # aleft=0
        # for i in range(0,height):
        #     for j in range(0,width):
        #         if result[i][j][0]!=0 or result[i][j][1]!=0 or result[i][j][2]!=0:
        #             # minwidth=min(minwidth,j)
        #             aleft+=j
        #             break
        # aleft/=height
        # awidth/=height
        # result=result[:,0:awidth]
        # result = cv2.warpPerspective(imageA, H, (awidth, imageA.shape[0]))
        # print "awidth ",awidth
        # print "aleft ",aleft
        # scale=float(((A[0]+B[0])-(C[0]+D[0]))/2)/float(imageA.shape[1])
        scale=float(max(A[0],B[0])-min(C[0],D[0]))/float(imageA.shape[1])
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

    def detectAndDescribe(self, image):
        # print("in detect and describe")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # descriptor = cv2.xfeatures2d.SIFT_create()
        descriptor = cv2.ORB_create()
        kps = descriptor.detect(image, None)
        kps, features = descriptor.compute(image, kps)
        kps = np.float32([kp.pt for kp in kps])
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
        # print("in matchkeypoints2")
        # computing a homography requires at least 4 matches
        # print len(matches)
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            # print("in matchkeypoints3")
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        # print("in matchkeypoints3 none")
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
    # global count=0
    stitcher = Stitcher()
    # imageA = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(1)+'.jpg')
    # imageB = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(6)+'.jpg')
    imageC = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(8)+'.jpg')
    imageD = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(2)+'.jpg')
    imageE = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(3)+'.jpg')
    imageF = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(4)+'.jpg')
    # imageG = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(5)+'.jpg')
    # imageH = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/Data/1/'+str(7)+'.jpg')
    # imageA = imutils.resize(imageA, width=400)
    # imageB = imutils.resize(imageB, width=400)
    imageC = imutils.resize(imageC, width=400)
    imageD = imutils.resize(imageD, width=400)
    imageE = imutils.resize(imageE, width=400)
    imageF = imutils.resize(imageF, width=400)
    # imageG = imutils.resize(imageG, width=400)
    # imageH = imutils.resize(imageH, width=400)
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    cv2.imshow("Image C", imageC)
    cv2.imshow("Image D", imageD)
    cv2.imshow("Image E", imageE)
    cv2.imshow("Image F", imageF)
    # cv2.imshow("Image G", imageG)
    # cv2.imshow("Image H", imageH)
    ans = stitcher.stitchleft([imageC, imageD], showMatches=True)
    (result1, vis) = ans
    cv2.imshow("Result1", result1)
    ans = stitcher.stitch([imageE, imageF], showMatches=True)
    (result2, vis) = ans
    cv2.imshow("Result2", result2)
    ans = stitcher.stitchleft([result1, result2], showMatches=True)
    (result3, vis) = ans
    cv2.imshow("Result3", result3)
    # cv2.imshow("Keypoint Matches1", vis)
    # cv2.imshow("Result1", result)
    # ans = stitcher.stitch([imageD, result], showMatches=True)
    # (result, vis) = ans
    # cv2.imshow("Result2", result)
    # cv2.imshow("Keypoint Matches2", vis)
    # ans = stitcher.stitch([imageC, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageD, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageC, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageB, result], showMatches=True)
    # (result, vis) = ans
    # ans = stitcher.stitch([imageA, result], showMatches=True)
    # (result, vis) = ans
    # cv2.imshow("Result3", result)
    # ans = stitcher.stitch([result, imageC], showMatches=True)
    # (result2, vis2) = ans
    # cv2.imshow("Keypoint Matches2", vis2)
    # cv2.imshow("Result2", result2)
    # ans = stitcher.stitch([result2, imageD], showMatches=True)
    # (result3, vis3) = ans
    # cv2.imshow("Keypoint Matches3", vis3)
    # cv2.imshow("Result3", result3)
    #show the images
    
    
    
    
    
    cv2.waitKey(0)
    # mapping={}
    # for i in range(1,9):
    #     max1=0
    #     max1j=0
    #     max2=0
    #     max2j=0
    #     for j in range(1,9):
    #         if i==j:
    #             continue
    #         imageA = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/InSample-20190907T082820Z-001/InSample/1/'+str(i)+'.jpg')
    #         imageB = cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/InSample-20190907T082820Z-001/InSample/1/'+str(j)+'.jpg')
    #         imageA = imutils.resize(imageA, width=400)
    #         imageB = imutils.resize(imageB, width=400)
    #         print(str(i)+" "+str(j))
    #         # stitch the images together to create a panorama
    #         stitcher = Stitcher()
    #         (kpsA, featuresA) = stitcher.detectAndDescribe(imageA)
    #         (kpsB, featuresB) = stitcher.detectAndDescribe(imageB)
    #         score=0
    #         M = stitcher.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0)
    #         if M is None:
    #             score=0
    #         else:
    #             (matches, H, status) = M
    #             score=len(matches)
    #         if score>max1:
    #             max2=max1
    #             max2j=max1j
    #             max1=score
    #             max1j=j
    #         elif score>max2:
    #             max2=score
    #             max2j=j
    #         # ans = stitcher.stitch([imageA, imageB], showMatches=True)
    #         # if ans is None:
    #         #     continue
    #         # (result, vis) = ans
    #         # show the images
    #         # cv2.imshow("Image A", imageA)
    #         # cv2.imshow("Image B", imageB)
    #         # cv2.imshow("Keypoint Matches", vis)
    #         # cv2.imshow("Result", result)
    #         # cv2.waitKey(0)
    #     mapping[i]=[[max1j,max1],[max2j,max2]]
    # print mapping
    # mini=9
    # minscore=1000
    # for i in range(1,9):
    #     if(mapping[i][0][1]<minscore):
    #         minscore=mapping[i][0][1]
    #         mini=i
    #     if(mapping[i][1][1]<minscore):
    #         minscore=mapping[i][1][1]
    #         mini=i
    # print mini,minscore
    # setting=set()
    # result=cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/InSample-20190907T082820Z-001/InSample/1/'+str(mini)+'.jpg')
    # setting.add(mini)
    # print str(mini)+" getting merged"
    # im=0
    # if(mapping[mini][0][1]!=minscore):
    #     im=mapping[mini][0][0]
    # else:
    #     im=mapping[mini][1][0]
    # count=0
    # while(1):
    #     count=count+1
    #     image=cv2.imread('/home/akshay/Sem1901/COL780/A2/Stitch-Panorama/InSample-20190907T082820Z-001/InSample/1/'+str(im)+'.jpg')
    #     setting.add(im)
    #     print str(im)+" getting merged"
    #     result = imutils.resize(result, width=400)
    #     image = imutils.resize(image, width=400)
    #     ans = stitcher.stitch([result, image], showMatches=True)
    #     if ans is None:
    #         break
    #     (result, vis) = ans
    #     cv2.imshow("Result"+str(count), result)
    #     if (mapping[im][0][0] in setting) and (mapping[im][1][0] in setting):
    #         break
    #     else:
    #         if (mapping[im][0][0] in setting):
    #             im=mapping[im][1][0]
    #         else:
    #             im=mapping[im][0][0]
    # cv2.imshow("Result"+str(count), result)
    
    # cv2.waitKey(0)