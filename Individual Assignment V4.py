
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  
import pytesseract #OCR TOOL, RGB form(opencv-BGR so need to transfer img first)

# import Tesseract.exe path
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract-OCR\tesseract.exe'

# load img
img_path = "G:\\Master\\CDS540_CV\\Ind_assgin\\Test Sample\\book.jpg"
IMG = cv.imread(img_path)

############### subfunction definition ###############
# display_plt
def img_display(titles,images,n):
    plt.figure(figsize=(16,8))

    for i in range(n):
        plt.subplot(2,4,i+1),plt.imshow(cv.cvtColor(images[i],cv.COLOR_BGR2RGB),extent=[0, 30, 0, 36])
        plt.title(titles[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# 1)resize
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[0:2] # get h and w from img data
    # original
    if width is None and height is None:
        return image
    # specific height
    if width is None:
        r = height / float(h) # get the ratio
        dim = (int(w * r), height)
    # specific width
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized

# 2) warp by using mouse
# mouse action callback function
pts=[]
def onMouseAction(event, x, y, flags, param):  # action--left click
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:  # left click
        pts.append(setpoint)  # add a click point to array
        print("point {}：{}".format(len(pts), setpoint))# display the chosen point

# select 4 points by mouse
def get_four_points(image):
    # choose 4 points
    print("pls left click to select 4 points :")
    status = True  # drawing status
    cv.namedWindow('origin') 
    cv.setMouseCallback('origin', onMouseAction, status)  # use the callback function to detect mouse action
    while True:
        if len(pts) > 0:
            cv.circle(image, pts[-1], 5, (0,0,255), -1)  # draw a dot 
        if len(pts) > 1:
            cv.line(image, pts[-1], pts[-2], (255, 0, 0), 2)  # draw a line
        if len(pts) == 4:  # while we have 4 points in array
            cv.line(image, pts[0], pts[-1], (255,0,0), 2)  # draw last line and leave
            cv.imshow('origin', image)
            cv.waitKey(1000)
            break
        cv.imshow('origin', image)
        cv.waitKey(100)
    cv.destroyAllWindows() 
    ptsSrc = np.array(pts)  #  (4,2)，Numpy 
    print(ptsSrc)
    
    return ptsSrc

# warp_find 4 points
def order_points(point):
    # define a matrix
    rect = np.zeros((4, 2), dtype='float32')
    s = point.sum(axis=1)
    rect[0] = point[np.argmin(s)]
    rect[2] = point[np.argmax(s)]
    diff = np.diff(point, axis=1)
    rect[1] = point[np.argmin(diff)]
    rect[3] = point[np.argmax(diff)]
    return rect

# warp_transfer
def four_point_transform(image, point):
    # organize the points in order
    rect = order_points(point)
    # top_left
    # top_right
    # bottom_right
    # bottom_left
    (tl, tr, br, bl) = rect
    # calculate the distance
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(heightA), int(heightB))
    # get the new point location
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype='float32')
    # calculate the matrix for warp
    M = cv.getPerspectiveTransform(rect, dst)
    # warp
    warped = cv.warpPerspective(image, M, (max_width, max_height))
    return warped

# get the warped image and turn it into gray
def Warp(image,point):
    Img_warped = four_point_transform(image, point.reshape(4, 2) * ratio)
    Scan = cv.cvtColor(Img_warped, cv.COLOR_BGR2GRAY)
    return Img_warped,Scan

# 3) mouse draw ROI rectangle
global image_roi
global drawing, top_left_pt, bottom_right_pt
drawing = False
def select_roi(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt,image_roi# change those global var inside this function
    if event == cv.EVENT_LBUTTONDOWN: # top left click point
        drawing = True # start drawing
        top_left_pt = (x, y) # get location

    elif event == cv.EVENT_LBUTTONUP: # bottom right release point
        drawing = False # close the drawing status
        bottom_right_pt = (x, y) # get the x,y location     
        cv.rectangle(image_roi, top_left_pt, bottom_right_pt, (0, 255, 0), 2) # draw rectangle
 
def nothing(x): #trackbar 回调 function none
    pass

#================= image Pre-processing subfunction=======================#

# 2) convert img to grayscale
def Gray(image):
    Img_Gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    return Img_Gray

# 3) Binary by Trackbar    
def Bianry_Trackbar(image):
    cv.namedWindow("Thresh_Binary")
    cv.createTrackbar("Binary_T","Thresh_Binary",120,255,nothing) # 120 起始值; 255 bar长度

    while True:
        Binary_T=cv.getTrackbarPos("Binary_T","Thresh_Binary") # get threshold value

        _,Img_Thresh=cv.threshold(image,Binary_T,255,cv.THRESH_BINARY)        
        cv.imshow("Binary Result",Img_Thresh)

        key = cv.waitKey(1) # Esc键
        if key == 27:
            break
    cv.destroyAllWindows()

    return Img_Thresh
    
# 4) guassian blur by trackbar
def GuassianBlur_Trackbar(image):
    cv.namedWindow("Gaussian Blur")
    cv.createTrackbar("Kernel Size", "Gaussian Blur", 1, 21, nothing)
    cv.createTrackbar("Standard Deviation", "Gaussian Blur", 1, 10, nothing)

    while True:
        GaussianBlur_size = cv.getTrackbarPos("Kernel Size", "Gaussian Blur")
        ksize = GaussianBlur_size * 2 +3
        GaussianBlur_sigma = cv.getTrackbarPos("Standard Deviation", "Gaussian Blur")
        std_dev = GaussianBlur_sigma/10.0

        Img_Blurred = cv.GaussianBlur(image,(ksize,ksize),std_dev)
        cv.imshow("Blurred Result",Img_Blurred)

        key = cv.waitKey(1) # Esc键
        if key == 27:
            break
    cv.destroyAllWindows()

    return Img_Blurred

# 5) find edge (Canny) (need to improve this part inorder to get better edge)
def Canny_Trackbar(image):
    cv.namedWindow("Canny Edge")
    cv.createTrackbar("Low_T","Canny Edge",50,400,nothing)
    cv.createTrackbar("High_T","Canny Edge",100,400,nothing)

    while True:
        L_Thresh=cv.getTrackbarPos("Low_T","Canny Edge")
        H_Thresh=cv.getTrackbarPos("High_T","Canny Edge")

        Img_Edge = cv.Canny(image, L_Thresh, H_Thresh)
        cv.imshow("Canny Result",Img_Edge)

        key = cv.waitKey(1) # Esc
        if key == 27:
            break
    cv.destroyAllWindows()

    return Img_Edge

# 6) find and draw all contours
def find_ALL_contours(image1,image2):

    Contours =  cv.findContours(image1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    Img_Contour_ALL = cv.drawContours(image2.copy(), Contours, -1, (0, 0, 255), 1)
    
    return Img_Contour_ALL,Contours

# 7) find and draw the largest contour
def find_Largest_Contour(Contours,image):
    # sort contours by area from large to small
    Contours = sorted(Contours, key=cv.contourArea, reverse=True)
    # iterate to find the largest contour 
    for c in Contours:
        # calculate perimeter
        perimeter = cv.arcLength(c, True)
        # get the closest peri value approximatly 
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
        # get the max peri
        if len(approx) == 4:
            screenCnt = approx
            break
    # draw the closest contour
    Img_Contour_MAX = cv.drawContours(image.copy(), [screenCnt], -1, (0, 0, 255), 1)

    return Img_Contour_MAX,screenCnt

# 8) warp 
def Warp(image,screenCnt):
    Img_warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
    Scan = cv.cvtColor(Img_warped, cv.COLOR_BGR2GRAY)
    return Img_warped,Scan

# 9) dialte
def Dilate_trackbar(image):
    cv.namedWindow("Dilate")
    cv.createTrackbar("Kernel","Dilate",1,10,nothing) 
    cv.createTrackbar("Iteration","Dilate",1,10,nothing)

    while True:
        ksize = cv.getTrackbarPos("Kernel","Dilate")
        kernel = np.ones((ksize,ksize),np.uint8)
        iteration = cv.getTrackbarPos("Iteration","Dilate")
        Img_dilated = cv.dilate(image,kernel,iterations=iteration)
        cv.imshow("dilate Result",Img_dilated)

        key = cv.waitKey(1) # Esc键
        if key == 27:
            break
    cv.destroyAllWindows()

    return Img_dilated

# 10) erode
def Erode_trackbar(image):
    cv.namedWindow("Erode")
    cv.createTrackbar("Kernel","Erode",1,10,nothing) 
    cv.createTrackbar("Iteration","Erode",1,10,nothing)

    while True:
        ksize = cv.getTrackbarPos("Kernel","Erode")
        kernel = np.ones((ksize,ksize),np.uint8)
        iteration = cv.getTrackbarPos("Iteration","Erode")
        Img_eroded = cv.erode(image,kernel,iterations=iteration)
        cv.imshow("Erode Result",Img_eroded)

        key = cv.waitKey(1) # Esc键
        if key == 27:
            break
    cv.destroyAllWindows()

    return Img_eroded
#================= OCR subfunction=======================#
def OCR(Scanned_File):
    # read and print text
    text = pytesseract.image_to_string(Scanned_File)
    print(text)
    # save the ocr result to txt file
    with open('ocr_result.txt', 'w') as f:
        print(text)
        f.write(str(text))
    
    # draw box
    scanned_file_copy = Scanned_File.copy()
    hImg, wImg, _ = Scanned_File.shape
    boxes = pytesseract.image_to_data(Scanned_File)
    print(boxes)
    for x,b in enumerate(boxes.splitlines()):
        if x!=0:
            b = b.split()
            print(b)
            if len(b)==12:
                x1, y1, x2, y2 = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv.rectangle(Scanned_File, (x1,y1), (x1+x2, y1+y2), (0, 0, 255), 3)
                cv.putText(Scanned_File, b[11], (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
    

#================= OCR Display subfunction=======================#

    #display result
    images_ocr = scanned_file_copy,Scanned_File
    titles_ocr = "scanned file","ocr result"
    plt.figure(figsize=(16,8))

    for i in range(2):            
        plt.subplot(1,2,i+1),plt.imshow(cv.cvtColor(images_ocr[i],cv.COLOR_BGR2RGB))
        plt.title(titles_ocr[i])
        plt.axis('off')
    plt.tight_layout()  # 自动调整子图之间的间距
    plt.show()
    

############### Main function ###############
ratio = IMG.shape[0] / 800.0
image_copy = IMG.copy()

# 1) resize
Img_Resized = resize(image_copy, height = 800)

# 2) image preprocessing
# warp
four_points = get_four_points(Img_Resized.copy())
Img_Warp,Scan = Warp(image_copy,four_points)

# 3)select ROI region
ROI_Img = resize(Img_Warp.copy(),height=800)
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1) # initialize the rectangle's tl pts and br pts
image_roi = ROI_Img
cv.namedWindow('ROI Selection') 
cv.setMouseCallback('ROI Selection', select_roi) # use callback

while True:
    cv.imshow('ROI Selection', ROI_Img)
    if cv.waitKey(1) == 27:  
        break

# get ROI
ROI_Img = ROI_Img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]] # image cutting: h，w,tl（0，0）
cv.imshow("ROI",ROI_Img)
cv.waitKey(0)

# 4) image preprocess
Img_Gray = Gray(ROI_Img)
Img_Thresh = Bianry_Trackbar(Img_Gray)
Img_eroded = Erode_trackbar(Img_Thresh)
Img_dilated = Dilate_trackbar(Img_eroded)
Img_Blurred = GuassianBlur_Trackbar(Img_dilated)
Img_Edge = Canny_Trackbar(Img_Blurred)

# 5) display final image after preprocess
Img_result = [IMG,Img_Warp,ROI_Img,Img_Gray,Img_Thresh,Img_dilated,Img_Blurred,Img_Edge]
img_titles = ["original image","Warpped image","ROI region","gray image","Binary image","eroded and dilated","gaussian blur","image edge"]
img_display(img_titles,Img_result,8)

# 6) save scanned file
_ = cv.imwrite('./scanned_file.jpg', Img_result[-2])

# 6) OCR the saved scanned file
scanned_file = cv.imread("./scanned_file.jpg")
OCR(scanned_file)

# 7)print the OCR file
with open("ocr_result.txt", "r") as file:
    print(file.read())



