import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# cap = cv2.VideoCapture('Tag0.mp4')
# i=0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)
#     cv2.imwrite('kang' + str(i) + '.jpg', frame)
#     i+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

img = cv2.imread("kang15.jpg", cv2.IMREAD_GRAYSCALE)
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
ret, thresh = cv2.threshold(img, 210, 255, 0)
contours=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
contours = contours[:len(contours)-1]
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Contours', img)
# cv2.waitKey(0)
print(contours)

epsilon = 0.1*cv2.arcLength(contours[1],True)
approx = cv2.approxPolyDP(contours[1],epsilon,True)

print(approx)
minx = sys.maxint
miny = sys.maxint
maxx = 0
maxy = 0
for c in approx:
    if minx > c[0][0]:
        minx = c[0][0]
    if miny > c[0][1]:
        miny = c[0][1]
    if maxx < c[0][0]:
        maxx = c[0][0]
    if maxy < c[0][1]:
        maxy = c[0][1]
cropped_image = img[miny:maxy, minx:maxx]
# cv2.imshow('cropped', cropped_image)
# cv2.waitKey(0)
inliers_src = []
for a in approx:
    inliers_src.append([a[0][1], a[0][0], 1])
# inliers_src = [[minx, miny, 1], [maxx, miny, 1], [minx, maxy, 1], [maxx, maxy, 1]]
inliers_dst = [[0,8], [8,8], [8,0],[0,0]]

A= []
for i in range(len(inliers_src)):
    A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0 ,0, inliers_src[i][0]*inliers_dst[i][0], inliers_src[i][1]*inliers_dst[i][0], inliers_dst[i][0]])
    A.append([ 0, 0 ,0,-inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0]*inliers_dst[i][1], inliers_src[i][1]*inliers_dst[i][1], inliers_dst[i][1]])
s, v, vh = np.linalg.svd(A)
H = vh[-1,:]
H = H.reshape((3,3))

print(H)
cords = []
for i in range(minx, maxx+1):
    for j in range(miny, maxy+1):
        cords.append([j,i,1])

print('initial cords', np.transpose(cords))
new_cords = np.matmul(H,np.transpose(cords))
print('transformed cords', np.transpose(new_cords))
nc = []
for n in range(new_cords.shape[1]):
    nc.append([new_cords[0][n]/new_cords[2][n], new_cords[1][n]/new_cords[2][n]])
print('simple h new cords ',nc)
new_image = np.zeros((8,8))
print(len(nc))
for i in range(len(nc)):
    if ( round(nc[i][0]) >= 0 and round(nc[i][1] >= 0) and round(nc[i][0]) < 8 and round(nc[i][1]) <8):
        print('thresh index', nc[i][0], nc[i][1])
        print('original cords', cords[i][0], cords[i][1])
        new_image[int(round(nc[i][0]))][int(round(nc[i][1]))] = thresh[cords[i][0]][cords[i][1]]
print(new_image)
identity = 0
theta = 0
if(new_image[2][2] == 255):
    identity = 1*(1 if new_image[4][4] == 255 else 0) + 2*(1 if new_image[4][3] == 255 else 0) + 4*(1 if new_image[3][3] == 255 else 0) + 8*(1 if new_image[3][4] == 255 else 0)
    theta = 2
elif (new_image[2][5] == 255):
    identity = 1*(1 if new_image[4][3] == 255 else 0) + 2*(1 if new_image[4][4] == 255 else 0) + 4*(1 if new_image[3][4] == 255 else 0) + 8*(1 if new_image[3][3] == 255 else 0)
    theta = 1
elif (new_image[5][5] == 255):
    identity = 1*(1 if new_image[3][3] == 255 else 0) + 2*(1 if new_image[3][4] == 255 else 0) + 4*(1 if new_image[4][4] == 255 else 0) + 8*(1 if new_image[4][3] == 255 else 0)
    theta = 0
elif (new_image[5][2] == 255):
    identity = 1*(1 if new_image[3][4] == 255 else 0) + 2*(1 if new_image[3][3] == 255 else 0) + 4*(1 if new_image[4][3] == 255 else 0) + 8*(1 if new_image[4][4] == 255 else 0)
    theta = 3

print('Identity is ', identity + 1)
print ('Coordinates are ', approx )
lena = cv2.imread("Lena.png", 1)
gray = cv2.imread("Lena.png", cv2.IMREAD_GRAYSCALE)
# scale_percent = 50 # percent of original size
# width = int(8)
# height = int(8)
# dim = (width, height)
# # resize image
# gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
inliers_dst = []
for a in approx:
    inliers_dst.append([a[0][1], a[0][0], 1])
# inliers_src = [[minx, miny, 1], [maxx, miny, 1], [minx, maxy, 1], [maxx, maxy, 1]]
if theta == 0:
    inliers_src = [[0,512], [512,512], [512,0],[0,0]]
elif theta == 1:
    inliers_src = [[0, 0], [0, 512], [512, 512], [512, 0]]
elif theta == 2:
    inliers_src = [[512, 0], [0, 0], [0, 512], [512, 512]]
elif theta == 3:
    inliers_src = [[512, 512], [512, 0], [0, 0], [0, 512]]

print('Source points', inliers_src)
print('Destination points', inliers_dst)

A= []
for i in range(len(inliers_src)):
    A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0 ,0, inliers_src[i][0]*inliers_dst[i][0], inliers_src[i][1]*inliers_dst[i][0], inliers_dst[i][0]])
    A.append([ 0, 0 ,0,-inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0]*inliers_dst[i][1], inliers_src[i][1]*inliers_dst[i][1], inliers_dst[i][1]])
s, v, vh = np.linalg.svd(A)
H = vh[-1,:]
H = H.reshape((3,3))
# theta = np.pi/2
print(theta)
# identity_matrix = np.identity(3)
# identity_matrix[0][0] = int(round(np.cos(theta)))
# identity_matrix[0][1] = int(round(-np.sin(theta)))
# identity_matrix[1][1] = int(round(np.cos(theta)))
# identity_matrix[1][0] = int(round(np.sin(theta)))
# print(identity_matrix)
# H = np.matmul(H, identity_matrix)

# identity_matrix = np.identity(3)
# identity_matrix[1][2] = maxy - miny
# identity_matrix[0][2] = maxx - minx
# print(identity_matrix)
# H = np.matmul(identity_matrix, H)
lena_cords = []
for i in range(gray.shape[1]):
    for j in range(gray.shape[0]):
        lena_cords.append([i,j,1])
# print('Lena cords', lena_cords)
lena_warped = np.matmul(H, np.transpose(lena_cords))
# print(lena_warped)
lena_warped_cords = []
for i in range(lena_warped.shape[1]):
    lena_warped_cords.append([int(round(lena_warped[0][i]/lena_warped[2][i])), int(round(lena_warped[1][i]/lena_warped[2][i]))])
# print(lena_warped_cords)
# print(img.shape)
for i in range(len(lena_warped_cords)):
    if (lena_warped_cords[i][0] >= 0 and lena_warped_cords[i][1] >= 0 and lena_warped_cords[i][0] < img.shape[0] and lena_warped_cords[i][1] < img.shape[1]):
        # print('thresh index', int(round(lena_warped_cords[i][0])), int(round(lena_warped_cords[i][1])))
        # print('original cords', lena_cords[i][0], lena_cords[i][1])
        img[lena_warped_cords[i][0]][lena_warped_cords[i][1]] = gray[lena_cords[i][0]][lena_cords[i][1]]
# print(theta)

cv2.imwrite('Lena_imposed.png', img)
# cv2.waitKey(0)



cv2.imwrite('transformed_image.png', new_image)
# cv2.imshow('image', thresh)
# cv2.waitKey(0)
#
# print(ptsWarped)
# print('new H', H_new)
#
# print(cropped_image)
# warped = cv2.warpPerspective(src=cropped_image, M=H_new, dsize=(8,8))
# warped_full = cv2.warpPerspective(src=img, M=H_new, dsize=(img.shape[1], img.shape[0]))
# # cv2.imshow('warped.png', warped)
# # cv2.imshow('warped_full.png', warped)
# # cv2.waitKey(0)
