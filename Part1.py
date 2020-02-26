import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

# cap = cv2.VideoCapture('Tag0.mp4')
# i=0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)
#     cv2.imwrite('multipleTags' + str(i) + '.jpg', frame)
#     i+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


def main():
    # img = cv2.imread("kang15.jpg", cv2.IMREAD_GRAYSCALE)
    img_array = []
    cap = cv2.VideoCapture('Tag0.mp4')
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    scale_percent = 60  # percent of original size
    frame_width = int(frame_width * scale_percent / 100)
    frame_height = int(frame_height * scale_percent / 100)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width,frame_height))
    
    while(True):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            scale_percent = 60 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            ret, thresh = cv2.threshold(img, 230, 255, 0)
            _, contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            # cv2.imshow('Contours', img)
            # cv2.waitKey(0)
            # print(contours)
            # print(hierarchy)
            index = findRelevantContours(hierarchy[0])
            # print(index)
            relContours = []
            threshold_area = 150
            for i in index:
                area = cv2.contourArea(contours[i])
                # print(area)
                if area > threshold_area:
                    relContours.append(contours[i])
            # cv2.drawContours(img, relContours, -1, (0, 255, 0), 3)
            # cv2.imshow('Relevant Contours', img)
            # cv2.waitKey(0)
            for contours in relContours:
                epsilon = 0.1 * cv2.arcLength(contours, True)
                approx = cv2.approxPolyDP(contours, epsilon, True)

                # print(approx)
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
                inliers_dst = [[0, 8], [8, 8], [8, 0], [0, 0]]

                A = []
                for i in range(len(inliers_src)):
                    A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0, 0, inliers_src[i][0] * inliers_dst[i][0],
                              inliers_src[i][1] * inliers_dst[i][0], inliers_dst[i][0]])
                    A.append([0, 0, 0, -inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0] * inliers_dst[i][1],
                              inliers_src[i][1] * inliers_dst[i][1], inliers_dst[i][1]])
                s, v, vh = np.linalg.svd(A)
                H = vh[-1, :]
                H = H.reshape((3, 3))

                # print(H)
                cords = []
                for i in range(minx, maxx + 1):
                    for j in range(miny, maxy + 1):
                        cords.append([j, i, 1])

                # print('initial cords', np.transpose(cords))
                new_cords = np.matmul(H, np.transpose(cords))
                # print('transformed cords', np.transpose(new_cords))
                nc = []
                for n in range(new_cords.shape[1]):
                    nc.append([new_cords[0][n] / new_cords[2][n], new_cords[1][n] / new_cords[2][n]])
                # print('simple h new cords ', nc)
                new_image = np.zeros((8, 8))
                # print(len(nc))
                for i in range(len(nc)):
                    if (round(nc[i][0]) >= 0 and round(nc[i][1] >= 0) and round(nc[i][0]) < 8 and round(nc[i][1]) < 8):
                        # print('thresh index', nc[i][0], nc[i][1])
                        # print('original cords', cords[i][0], cords[i][1])
                        new_image[int(round(nc[i][0]))][int(round(nc[i][1]))] = thresh[cords[i][0]][cords[i][1]]
                # print(new_image)
                identity = 0
                theta = 0
                if (new_image[2][2] == 255):
                    identity = 1 * (1 if new_image[4][4] == 255 else 0) + 2 * (1 if new_image[4][3] == 255 else 0) + 4 * (
                        1 if new_image[3][3] == 255 else 0) + 8 * (1 if new_image[3][4] == 255 else 0)
                    theta = 2
                elif (new_image[2][5] == 255):
                    identity = 1 * (1 if new_image[4][3] == 255 else 0) + 2 * (1 if new_image[4][4] == 255 else 0) + 4 * (
                        1 if new_image[3][4] == 255 else 0) + 8 * (1 if new_image[3][3] == 255 else 0)
                    theta = 1
                elif (new_image[5][5] == 255):
                    identity = 1 * (1 if new_image[3][3] == 255 else 0) + 2 * (1 if new_image[3][4] == 255 else 0) + 4 * (
                        1 if new_image[4][4] == 255 else 0) + 8 * (1 if new_image[4][3] == 255 else 0)
                    theta = 0
                elif (new_image[5][2] == 255):
                    identity = 1 * (1 if new_image[3][4] == 255 else 0) + 2 * (1 if new_image[3][3] == 255 else 0) + 4 * (
                        1 if new_image[4][3] == 255 else 0) + 8 * (1 if new_image[4][4] == 255 else 0)
                    theta = 3

                # print('Identity is ', identity + 1)
                # print ('Coordinates are ', approx)
                # lena = cv2.imread("Lena.png", 1)
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
                    inliers_src = [[0, 512], [512, 512], [512, 0], [0, 0]]
                elif theta == 1:
                    inliers_src = [[0, 0], [0, 512], [512, 512], [512, 0]]
                elif theta == 2:
                    inliers_src = [[512, 0], [0, 0], [0, 512], [512, 512]]
                elif theta == 3:
                    inliers_src = [[512, 512], [512, 0], [0, 0], [0, 512]]

                # print('Source points', inliers_src)
                # print('Destination points', inliers_dst)

                A = []
                for i in range(len(inliers_src)):
                    A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0, 0, inliers_src[i][0] * inliers_dst[i][0],
                              inliers_src[i][1] * inliers_dst[i][0], inliers_dst[i][0]])
                    A.append([0, 0, 0, -inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0] * inliers_dst[i][1],
                              inliers_src[i][1] * inliers_dst[i][1], inliers_dst[i][1]])
                s, v, vh = np.linalg.svd(A)
                H = vh[-1, :]
                H = H.reshape((3, 3))
                # theta = np.pi/2
                # print(theta)
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
                        lena_cords.append([i, j, 1])
                # print('Lena cords', lena_cords)
                lena_warped = np.matmul(H, np.transpose(lena_cords))
                # print(lena_warped)
                lena_warped_cords = []
                for i in range(lena_warped.shape[1]):
                    lena_warped_cords.append(
                        [int(round(lena_warped[0][i] / lena_warped[2][i])), int(round(lena_warped[1][i] / lena_warped[2][i]))])
                # print(lena_warped_cords)
                # print(img.shape)
                for i in range(len(lena_warped_cords)):
                    if (lena_warped_cords[i][0] >= 0 and lena_warped_cords[i][1] >= 0 and lena_warped_cords[i][0] < img.shape[0] and
                            lena_warped_cords[i][1] < img.shape[1]):
                        # print('thresh index', int(round(lena_warped_cords[i][0])), int(round(lena_warped_cords[i][1])))
                        # print('original cords', lena_cords[i][0], lena_cords[i][1])
                        img[lena_warped_cords[i][0]][lena_warped_cords[i][1]] = gray[lena_cords[i][0]][lena_cords[i][1]]
                # print(theta)

                # cv2.imwrite('Lena_imposed.png', img)
                # cv2.waitKey(0)

                # cv2.imwrite('transformed_image.png', new_image)
            # print('img', img.shape)
            # print('out', frame_width,frame_height)
            # out.write(img)

            # cv2.imshow('frame',img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_array.append(img)
            # cv2.imwrite('multipleTags' + str(i) + '.jpg', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cap.release()
    # out.release()
    # cv2.destroyAllWindows()



def findRelevantContours(hierarchy):
    index = []
    for i,hier in enumerate(hierarchy):
        # print(hier)
        if hier[2] != -1 and hier[3] != -1:
            index.append(i)
    return index


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


if __name__ == '__main__':
    main()

def drawCube(inliers_dst, img, img_copy):
    # Compute H for cube
    inliers_src = [[0, 1], [1, 1], [1, 0], [0,0]]
    print('Source points', inliers_src)
    print('Destination points', inliers_dst)

    A= []
    for i in range(len(inliers_src)):
        A.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0 ,0, inliers_src[i][0]*inliers_dst[i][0], inliers_src[i][1]*inliers_dst[i][0], inliers_dst[i][0]])
        A.append([ 0, 0 ,0,-inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0]*inliers_dst[i][1], inliers_src[i][1]*inliers_dst[i][1], inliers_dst[i][1]])
    s, v, vh = np.linalg.svd(A)
    H = vh[-1,:]
    H = H.reshape((3,3))
    print("xxxxxxxxxxxxxxxxxxxxxx")
    print(H)

    # Camera Parameters
    K = np.matrix.transpose(np.asarray([[1406.08415449821, 0, 0],
                                        [2.20679787308599, 1417.99930662800, 0],
                                        [1014.13643417416, 566.347754321696, 1]]))
    print("K: ")
    print(K)

    K_inv = np.linalg.inv(K)
    print("K_inv: ")
    print(K_inv)

    B_ = np.matmul(K_inv, H)
    print("B_: ")
    print(B_)

    if (np.linalg.det(B_) < 0):
        B_ = (-1) * B_

    print(B_)

    # Get colums of B_
    x1 = B_[:,0]
    x2 = B_[:,1]
    print(x1)
    print(x2)

    # x3 = np.cross(x1, x2)
    y = B_[:,2]
    print(y)

    # Calculate scale factor
    scale = ((np.linalg.norm(x1) + np.linalg.norm(x2))/2)**(-1)
    print("scale: ")
    print(scale)

    # Get the rotation and translation vectors
    r1 = scale * x1
    r2 = scale * x2
    # r3 = scale * x3
    r3 = np.cross(r1, r2)
    t = scale * y

    # print(r1)
    r1 = np.reshape(r1, (3,1))
    r2 = np.reshape(r2, (3,1))
    r3 = np.reshape(r3, (3,1))
    t = np.reshape(t, (3,1))


    # print(r1)

    # Projection Matrix
    P = np.asarray([r1, r2, r3, t])
    P = np.reshape(P, (3, 4))
    print("P: ")
    print(P)

    # Coordinates of corner of cube in world frame (Homogeneous Coordinates)
    cube_w = np.asarray([[0, 1, 1, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1]])

    # Coordinates of corner of cube in image plane (Homogeneous Coordinates)
    cube_i = np.matmul(P, cube_w)
    print(cube_i)
    for i in range(len(cube_i[0])):
        alpha = cube_i[2][i]
        cube_i[0][i] = int(cube_i[0][i]/alpha)
        cube_i[1][i] = int(cube_i[1][i]/alpha)
        cube_i[2][i] = int(cube_i[2][i]/alpha)
        print(cube_i[:,i])
    print("cibe_i: ")
    print(cube_i)

    # c_line contains the pair of verties that have an edge between them
    c_lines = [[0, 1],
               [0, 4],
               [0, 3],
               [1, 5],
               [1, 2],
               [2, 6],
               [2, 3],
               [3, 7],
               [4, 5],
               [4, 7],
               [5, 6],
               [6, 7]]

    for i in range(len(cube_i)):
        pt1 = (int(cube_i[0][c_lines[i][0]]), int(cube_i[1][c_lines[i][0]]))
        pt2 = (int(cube_i[0][c_lines[i][1]]), int(cube_i[1][c_lines[i][1]]))
        image_copy = cv2.line(img_copy, pt1, pt2, (0, 0, 225), 2)

    cv2.imwrite('Cube_imposed.png', img)
