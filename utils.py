from imutils import paths
import cv2
import numpy as np

def loadImages_Duy(path, resize):
    """Load Images from path to array, @param path is the folder which containing images, @param resize is True
    if image is halved in size, otherwise is False"""

    image = cv2.imread(path)
    if resize == 1:
        image = cv2.resize(
            image, (int(image.shape[1] / 4), int(image.shape[0] / 4))
        )
    return image

def trim(frame):
    """crop frame """
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop bottom
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop left
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop right
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def padding(img, top, bottom, left, right):
    """add padding to img"""
    border = cv2.copyMakeBorder(
        img,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
    )
    return border

def convertResult(img):
    '''Because of your images which were loaded by opencv, 
    in order to display the correct output with matplotlib, 
    you need to reduce the range of your floating point image from [0,255] to [0,1] 
    and converting the image from BGR to RGB:'''
    img = np.array(img,dtype=float)/float(255)
    img = img[:,:,::-1]
    return img