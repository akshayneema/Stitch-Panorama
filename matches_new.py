import numpy as np
import cv2
import sys
from matchers import matchers
import time

class Stitch:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
		print filenames
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()

	def prepare_lists(self):
		print "Number of images : %d"%self.count
		self.centerIdx = self.count/2 
		print "Center index image : %d"%self.centerIdx
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print "Image lists prepared"

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		i = 0
		for b in self.left_list[1:]:
			
			cv2.imshow("image A" + str(i), a)
			# cv2.imwrite("leftimage" + str(i) + ".jpg", a)
			i = i + 1
			cv2.imshow("image B", b)
			print "a shape => ",a.shape
			print "b shape => ",b.shape
			H = self.matcher_obj.match(a, b, 'left')
			print "Homography is : ", H
			xh = self.matcher_obj.match(b, a, 'left')
			print "Inverse Homography :", xh
			# without changing the H
			a_dash = cv2.warpPerspective(a, H, (a.shape[1]*2,a.shape[0]*2))
			cv2.imshow("basic: ", a_dash )
			cv2.waitKey()

			# start point lower
			f1 = np.dot(H, np.array([0,0,1]))
			f1 = f1/f1[-1]
			print "f1 => ",f1
			# find offset of start point
			offsety1 = abs(int(f1[1]))
			offsetx1 = abs(int(f1[0]))
			print "offset start upper => ",(offsetx1,offsety1)

			# start lower point
			f2 = np.dot(H, np.array([0, a.shape[0], 1]))
			f2 = f2/f2[-1]
			print "f2 => ",f2
			# offset of start point lower
			offsety2 = abs(int(f2[1])) - a.shape[0]
			offsetx2 = abs(int(f2[0]))
			print "offset start lower => ",(offsetx2,offsety2)
			
			# end point upper
			f3 = np.dot(H, np.array([a.shape[1],0,1]))
			f3 = f3/f3[-1]
			print "f3 => ",f3
			# find offset of start point
			offsety3 = abs(int(f3[1]))
			offsetx3 = abs(int(f3[0]))
			print "offset start upper => ",(offsetx3,offsety3)
			
			# end point lower
			f4 = np.dot(H, np.array([a.shape[1],a.shape[0],1]))
			f4 = f4/f4[-1]
			print "f4 => ",f4
			# find offset of start point
			offsety4 = abs(int(f4[1]))
			offsetx4 = abs(int(f4[0]))
			print "offset start upper => ",(offsetx4,offsety4)
			
			# change matrix
			T = np.identity(3)
			T[0][-1] = -1*f1[0]
			T[1][-1] = -1*f1[1]
			H = np.dot(T,H)
			# H[0][-1] += abs(f1[0])
			# H[1][-1] += abs(f1[1])

			# end point
			ds = np.dot(H, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds/ds[-1]
			print "final ds =>", ds
			
			# make dsize
			dsize = (b.shape[1] + offsetx1, b.shape[0] + offsety1 + offsety2)
			print "image dsize =>", dsize
			# make transformation
			tmp = cv2.warpPerspective(a, H, dsize)
			# Stitch
			print "tmp shape => ",tmp.shape
			print "x coord => ",((b.shape[0]+offsety1),offsety1,(b.shape[0]+offsety1) - offsety1)
			print "y coord => ",((b.shape[1]+offsetx1),offsetx1,(b.shape[1]+offsetx1) - offsetx1)
			tmp[offsety1: (b.shape[0] + offsety1), offsetx1:(b.shape[1] + offsetx1)] = b

			scale = (f3[0] - f1[0])/a.shape[1]
			print scale
			tmp[:,int(abs(f1[0]) - abs(f1[0])/scale): int(abs(f1[0]))] = cv2.resize(tmp[:,0:int(abs(f1[0]))], (-int(abs(f1[0]) - abs(f1[0])/scale)+int(abs(f1[0])),tmp.shape[0]), interpolation = cv2.INTER_AREA)
			tmp=tmp[:,int(abs(f1[0]) - abs(f1[0])/scale):]
			a = tmp
			# cv2.resize(a,(480, 320))
			# print "new a shape => ",(a.shape)
			cv2.imshow("final a: ", a)
			cv2.waitKey()

		cv2.waitKey()
					
		self.leftImage = tmp

		
	def rightshift(self):
		for each in self.right_list:
			cv2.imshow("image leftImage", self.leftImage)
			cv2.imshow("image each", each)
			H = self.matcher_obj.match(each, self.leftImage, 'right')
			print "Homography :", H
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (max(int(txyz[0]),self.leftImage.shape[1]), max(int(txyz[1]),self.leftImage.shape[0]))
			tmp = cv2.warpPerspective(each, H, dsize)
			cv2.imshow("right basics => ",tmp)
			cv2.waitKey(0)
			# tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
			tmp = self.mix_and_match(self.leftImage, tmp)
			print "tmp shape",tmp.shape
			print "self.leftimage shape=", self.leftImage.shape
			self.leftImage = tmp
		# self.showImage('left')



	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print leftImage[-1,-1]

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print time.time() - t
		print black_l[-1]

		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
		return warpedImage




	def trim_left(self):
		pass

	def showImage(self, string=None):
		if string == 'left':
			cv2.imshow("left image", self.leftImage)
			# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
		elif string == "right":
			cv2.imshow("right Image", self.rightImage)
		cv2.waitKey()


if __name__ == '__main__':
	try:
		args = sys.argv[1]
	except:
		args = "./seq_1.txt"
		# args = "./seq_2.txt"
		# args = "./seq_3.txt"
		# args = "./seq_4.txt"
		# args = "./seq_5.txt"
	finally:
		print "Parameters : ", args
	s = Stitch(args)
	s.leftshift()
	cv2.imshow("leftimage.jpg", s.leftImage)
	s.showImage('left')
	s.rightshift()
	cv2.imshow("final.jpg", s.leftImage)
	print "done"
	cv2.imwrite("test12.jpg", s.leftImage)
	print "image written"
	cv2.destroyAllWindows()