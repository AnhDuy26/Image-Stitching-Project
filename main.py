from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageTk,Image
import os
import math
import utils

# ================== FIND AND MATCHING FEATURES ==================

def findAndDescribeFeatures(image, opt="ORB"):
    """find and describe features of @image,
        if opt='SURF', SURF algorithm is used.
        if opt='SIFT', SIFT algorithm is used.
        if opt='ORB', ORB algorithm is used.
        @Return keypoints and features of img"""
    # Getting gray image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if opt == "SURF":
        md = cv2.xfeatures2d.SURF_create()
    if opt == "ORB":
        md = cv2.ORB_create(nfeatures=3000)
    if opt == "SIFT":
        md = cv2.xfeatures2d.SIFT_create()
    # Find interest points and Computing features.
    keypoints, features = md.detectAndCompute(grayImage, None)
    # Converting keypoints to numbers.
    # keypoints = np.float32(keypoints)
    features = np.float32(features)
    return keypoints, features


def matchFeatures(featuresA, featuresB, ratio=0.75, opt="FB"):
    """matching features beetween 2 @features.
         If opt='FB', FlannBased algorithm is used.
         If opt='BF', BruteForce algorithm is used.
         @ratio is the Lowe's ratio test.
         @return matches"""
    if opt == "BF":
        featureMatcher = cv2.DescriptorMatcher_create("BruteForce")
    if opt == "FB":
        # featureMatcher = cv2.DescriptorMatcher_create("FlannBased")
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        featureMatcher = cv2.FlannBasedMatcher(index_params, search_params)

    # performs k-NN matching between the two feature vector sets using k=2
    # (indicating the top two matches for each feature vector are returned).
    matches = featureMatcher.knnMatch(featuresA, featuresB, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) > 4:
        return good
    thongbao.delete('1.0', END)
    thongbao.insert(INSERT, 'Please try again with another image set !')
    thongbao.configure(fg='red')
    hinh_anh.configure(image = '')
    raise Exception("Not enought matches")


def generateHomography(src_img, dst_img, ransacRep=5.0):
    """@Return Homography matrix, @param src_img is the image which is warped by homography,
        @param dst_img is the image which is choosing as pivot, @param ratio is the David Lowe’s ratio,
        @param ransacRep is the maximum pixel “wiggle room” allowed by the RANSAC algorithm
        """

    src_kp, src_features = findAndDescribeFeatures(src_img)
    dst_kp, dst_features = findAndDescribeFeatures(dst_img)

    good = matchFeatures(src_features, dst_features)

    src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacRep)
    matchesMask = mask.ravel().tolist()
    return H, matchesMask


def drawKeypoints(img, kp):
    img1 = img
    cv2.drawKeypoints(img, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img1


def drawMatches(src_img, src_kp, dst_img, dst_kp, matches, matchesMask):
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask[:100],  # draw only inliers
        flags=2,
    )
    return cv2.drawMatches(
        src_img, src_kp, dst_img, dst_kp, matches[:100], None, **draw_params
    )

# ================== STITCHING AND BLENDING ==================

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window / 2)
    try:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])


def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side, showstep=False):
    """Given two aligned images @dst_img and @src_img_warped, and the @width_dst is width of dst_img
    before resize, that indicates where there is the discontinuity between the images,
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one"""

    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)
    mask1 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True
    )
    mask2 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False
    )

    if showstep:
        nonblend = src_img_warped + dst_img_rz
    else:
        nonblend = None
        leftside = None
        rightside = None

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        pano = cv2.flip(pano, 1)
        if showstep:
            leftside = cv2.flip(src_img_warped, 1)
            rightside = cv2.flip(dst_img_rz, 1)
    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        if showstep:
            leftside = dst_img_rz
            rightside = src_img_warped

    return pano, nonblend, leftside, rightside


def warpTwoImages(src_img, dst_img, showstep=False):

    # generate Homography matrix
    H, _ = generateHomography(src_img, dst_img)

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32(
        [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
    ).reshape(-1, 1, 2)

    try:
        # aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        # find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        # otherwise side=right
        # source image is merged to the left side or right side of destination image
        if pts[0][0][0] < 0:
            side = "left"
            width_pano = width_dst + t[0]
        else:
            width_pano = int(pts1_[3][0][0])
            side = "right"
        height_pano = ymax - ymin

        # Translation
        # https://stackoverflow.com/a/20355545
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped = cv2.warpPerspective(
            src_img, Ht.dot(H), (width_pano, height_pano)
        )
        # generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz = np.zeros((height_pano, width_pano, 3))
        if side == "left":
            dst_img_rz[t[1] : height_src + t[1], t[0] : width_dst + t[0]] = dst_img
        else:
            dst_img_rz[t[1] : height_src + t[1], :width_dst] = dst_img

        # blending panorama
        pano, nonblend, leftside, rightside = panoramaBlending(
            dst_img_rz, src_img_warped, width_dst, side, showstep=showstep
        )

        # croping black region
        pano = crop(pano, height_dst, pts)
        return pano, nonblend, leftside, rightside
    except BaseException:
        thongbao.delete('1.0', END)
        thongbao.insert(INSERT, 'Please try again with another image set !')
        thongbao.configure(fg='red')
        hinh_anh.configure(image = '')
        raise Exception("Please try again with another image set!")

def multiStitching(list_images):
    """assume that the list_images was supplied in left-to-right order, choose middle image then
    divide the array into 2 sub-arrays, left-array and right-array. Stiching middle image with each
    image in 2 sub-arrays. @param list_images is The list which containing images, @param smoothing_window is
    the value of smoothy side after stitched, @param output is the folder which containing stitched image
    """
    n = int(len(list_images) / 2 + 0.5)
    left = list_images[:n]
    right = list_images[n - 1 :]
    right.reverse()
    while len(left) > 1:
        dst_img = left.pop()
        src_img = left.pop()
        left_pano, _, _, _ = warpTwoImages(src_img, dst_img)
        left_pano = left_pano.astype("uint8")
        left.append(left_pano)

    while len(right) > 1:
        dst_img = right.pop()
        src_img = right.pop()
        right_pano, _, _, _ = warpTwoImages(src_img, dst_img)
        right_pano = right_pano.astype("uint8")
        right.append(right_pano)

    # if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
    if right_pano.shape[1] >= left_pano.shape[1]:
        fullpano, _, _, _ = warpTwoImages(left_pano, right_pano)
    else:
        fullpano, _, _, _ = warpTwoImages(right_pano, left_pano)
    return fullpano


def crop(panorama, h_dst, conners):
    """crop panorama based on destination.
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and
    4 conners of destination image"""
    # find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    # conners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value<0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : h_dst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
    return panorama

def twoStitching(list_images):
    k0,f0=findAndDescribeFeatures(list_images[0],opt='ORB')
    k1,f1=findAndDescribeFeatures(list_images[1],opt='ORB')
    #matching features using BruteForce 
    mat= matchFeatures(f0,f1,ratio=0.6,opt='BF')
    #Computing Homography matrix and mask
    H,matMask= generateHomography(list_images[0],list_images[1])
    #wrap 2 image
    #choose list_images[0] as desination
    pano,non_blend,left_side,right_side= warpTwoImages(list_images[1],list_images[0],True)
    return pano

# ================== FUNCTIONS FOR GUI ==================

def dangnhap(username):
    # dirFile = username.get()
    dirFile = username
    list_images = []
    for img in dirFile:
        img_new = utils.loadImages_Duy(img, resize=0)
        list_images.append(img_new)
    # list_images.sort()
    # list_images= loadImages(dirFile,resize=0)
    # print(list_images)
    if len(list_images) < 2:
        thongbao.delete('1.0', END)
        thongbao.insert(INSERT, 'You must choose at least 2 images !')
        thongbao.configure(fg='red')
        hinh_anh.configure(image = '')
        return
    elif len(list_images) == 2:
        panorama = twoStitching(list_images)
    else:
        panorama= multiStitching(list_images)
    
    WIDTH = 1300
    HEIGHT = 275
    cv2.imwrite('result.jpg', panorama)
    thongbao.delete('1.0', END)
    thongbao.insert(INSERT, 'Successful stitching !')
    thongbao.configure(fg='green')
    img_import = (Image.open(r'result.jpg'))
    resize_w = img_import.size[0]
    resize_h = img_import.size[1]
    # resize = img_import.resize((resize_w // 7,resize_h // 7), Image.ANTIALIAS)
    if resize_w <= WIDTH and resize_h <= HEIGHT:
        resize = img_import
    elif resize_w > WIDTH and resize_h <= HEIGHT:
        coeff = math.ceil(resize_w / WIDTH) 
        resize = img_import.resize((resize_w // coeff,resize_h // coeff), Image.ANTIALIAS)
    elif resize_w <= WIDTH and resize_h > HEIGHT:
        coeff = math.ceil(resize_h / HEIGHT) 
        resize = img_import.resize((resize_w // coeff,resize_h // coeff), Image.ANTIALIAS)
    else:
        co1 = math.ceil(resize_w / WIDTH)
        co2 = math.ceil(resize_h / HEIGHT)
        if co1 >= co2:
            resize = img_import.resize((resize_w // co1,resize_h // co1), Image.ANTIALIAS)
        else:
            resize = img_import.resize((resize_w // co2,resize_h // co2), Image.ANTIALIAS)
    # resize = img_import.resize((1300,275), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(resize)
    hinh_anh.configure(image = img)
    hinh_anh.image = img

def upload_file(ROOF):
    # for label in list_label: label.destroy()
    f_types = [('JPG Files', '*.jpg')]   # type of files to select 
    filename = filedialog.askopenfilename(multiple=True,filetypes=f_types)
    # pathname = os.path.dirname(filename[0])
    x_coor = 50
    y_coor = 250
    global list_label
    list_label = []
    # print(filename)
    for f in filename:
        img=Image.open(f) # read the image file
        img=img.resize((150,150)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =Label(ROOF)
        list_label.append(e1)
        # e1.grid(row=row,column=col)
        e1.place(x = x_coor, y = y_coor)
        e1.image = img
        e1['image']=img # garbage collection 
        if(x_coor==1000): # start new line after third column
            y_coor=y_coor+200# start wtih next row
            x_coor=150   # start with first column
        else:       # within the same row 
            x_coor=x_coor+150 # increase to next column
    return filename

def dangnhap1(ROOF):
    username = upload_file(ROOF)
    dangnhap(username)
    # openResultWindow()

def clearImage():
    for label in list_label: label.destroy()
    hinh_anh.configure(image = '')

def openGuideWindow(ROOF):
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(ROOF)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("Guideline")
 
    # sets the geometry of toplevel
    newWindow.geometry("515x200")
 
    # A Label widget to show in toplevel
    guidelineText = "1. Click \"Upload and Stitch Image\" button to do the stitching.\n2. Click \"Clear\" button to clear image cache.\n3. Do the same process if you want to stitching another set.\nNote: \n- Your image set must be named from left to right\n- The result image will be saved in the same folder with your program."
    Label(newWindow, text =guidelineText, font = ('Time New Roman', 13),anchor="e", justify=LEFT).pack()
# ========================
# End of function
# ========================

def main():
    global ROOF
    ROOF = Tk()
    ROOF.iconbitmap("cokcok.ico")
    ROOF.state('zoomed')
    ROOF.title('IMAGE STITCHING BY GROUP 1')
    ROOF.configure(bg='lavender')
    # ========================
    # ========================
    image = Image.open('bg1.jpg')
    copy_of_image = image.copy()
    photo = ImageTk.PhotoImage(image)
    canvas = Canvas(ROOF, width = 1000,height = 600)
    canvas.pack(fill='both', expand = True)
    canvas.create_image(0, 0, image=photo,anchor = "nw")
    canvas.create_text(685, 40, text = 'PANORAMA',fill="white",font=('Castellar', 35, 'bold'))

    
    global hinh_anh
    hinh_anh = Label(ROOF, image = '', bg='black')
    hinh_anh.place(x = 50, y = 420)

    global thongbao
    thongbao = Text(ROOF, height = 1, width=50, font = ('Time New Roman', 20))
    thongbao.place(x = 430, y = 90)

    Button(ROOF, text='Upload and Stitch Image',font = ('Time New Roman', 20), bg='skyblue',width=20, height=1, command=lambda:dangnhap1(ROOF)).place(x = 20, y = 80)
    # hinh_anh = Label(ROOF, text = 'Result image', image = None)


    clearButton = Button(ROOF, text='Clear',font = ('Time New Roman', 20), bg='skyblue',width=10, height=1, command=clearImage).place(x = 20, y = 150)
    guideButton = Button(ROOF, text='Guideline',font = ('Time New Roman', 20), bg='skyblue',width=10, command=lambda:openGuideWindow(ROOF)).place(x= 200, y = 150)
    
    ROOF.mainloop()

if __name__ == '__main__':
     main()