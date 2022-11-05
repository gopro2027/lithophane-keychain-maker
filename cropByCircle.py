import cv2
import numpy as np
import math

mouseX = 0
mouseY = 0
img_width = 0
img_height = 0
cropRadius = 0

def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def isPointInsideCircle(x,y, cx,cy,radius):
    d = dist(x,y,cx,cy)
    if d <= radius:
        return True
    return False

def eraseCircle(img,x,y,radius, lineWidth):
    img2 = np.zeros((int(radius*2),int(radius*2),4), np.uint8)
    img2[:,0:int(radius*2)] = (255,255,255,0)      # (A, B, G, R) ?

    radius = int(radius)

    r, g, b, aa = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    #gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) * aa / 255.0

    for iy in range(- radius, radius):
        dx = int(math.sqrt(radius * radius - iy * iy))
        for ix in range(- dx,dx):
            try :
                a = float(float(aa[y + iy,x + ix])/255.0) #img[y + iy,x + ix,3]
                #gray = (0.2989 * img[y + iy,x + ix,0] + 0.5870 * img[y + iy,x + ix,1] + 0.1140 * img[y + iy,x + ix,2]) * a
                #if a != 0:
                    #print(a)'''or gray[y + iy,x + ix] < 0.5'''
                if a > 0.05: #spot is not transparent, only want to copy solid parts
                    img2[iy+radius, radius+ix] = img[y + iy, x + ix]
                else: #copy over solid white so the inside so keyrings work earlier and it matches the other areas
                    img2[iy+radius, radius+ix] = (255,255,255,255)
                if not isPointInsideCircle(y + iy, x + ix, y, x, radius-lineWidth):
                    img2[iy+radius, radius+ix] = (0,0,0,255)
            except:
                #print("Error!")
                fdya = 0 #literally nothing
    #exit(0)
    return img2


def create_keyring(img, x,y):
    global cropRadius,img_width,img_height, g_circleWidthMM, g_circleRingWidthMM
    #img_write = img.copy() #will draw on top of this one
    #cv2.circle(img,(x,y),int(holeSizePixel),(255,0,0,255),int(holeWallSizePixel),2) #ring
    #cv2.circle(img,(x,y),int(holeSizePixel),(255,0,0,255),int(holeWallSizePixel),2) #hole
    #cv2.circle(img_write,(x,y),int(holeSizePixel),(0,0,0,0),-1) #center

    #eraseCircle(img,x,y,holeWallSizePixel,[255,0,0,255] if tmpRing else [0,0,0,255],True)
    #eraseCircle(img,x,y,holeSizePixel,[0,0,0,0],False)

    lineWidth = (cropRadius*2 / g_circleWidthMM) * g_circleRingWidthMM / 2

    return eraseCircle(img,x,y,cropRadius, lineWidth)

def mouse_event(event,x,y,flags,param):
    global mouseX, mouseY, img, hasSelectedHole, output_img
    mouseX = x
    mouseY = y
    if event != 0:
        print("event: "+str(event))

    if event == 1: # 1 is the actual click button based on output    cv2.EVENT_LBUTTONDBLCLK is 7:
        print("clicked!")
        output_img = create_keyring(img, x,y)
        hasSelectedHole = True

def openImageForKeyring(img_open,circleWidthMM,circleRingWidthMM):
    global img, img_orig, mouseX, mouseY, cropRadius,img_width,img_height, hasSelectedHole, output_img, g_circleWidthMM, g_circleRingWidthMM
    g_circleWidthMM = circleWidthMM
    g_circleRingWidthMM = circleRingWidthMM

    hasSelectedHole = False
    img_orig = cv2.imread(img_open, cv2.IMREAD_UNCHANGED)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2RGBA)

    height, width, channels = img_orig.shape
    img_width = width
    img_height = height

    img_blank = np.zeros((width*2,width*2,4), np.uint8)
    img_blank[:,0:width*2] = (255,255,255,255)

    y_offset = int(width*2/4)
    x_offset = int(width*2/4)

    img_blank[y_offset:y_offset+img_orig.shape[0], x_offset:x_offset+img_orig.shape[1]] = img_orig


    img = img_blank.copy()

    cropRadius = img_width/2

    print("press q to cancel image editing")
    print("press r to remove circle crop")
    print("press s to save current image")
    print("left click to place circle crop")
    print('press k and l to change size of crop circle')
    #cv2.resizeWindow('image', width, height)
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image',mouse_event)
    while(1):
        if hasSelectedHole == True:
            return output_img
        img_tmp = img.copy()
        #img_tmp = create_keyring(img_tmp,mouseX,mouseY)
        lineWidth = (cropRadius*2 / g_circleWidthMM) * g_circleRingWidthMM
        cv2.circle(img_tmp,(mouseX,mouseY),int(cropRadius),(255,0,0,255),int(lineWidth))
        #img2 = np.zeros((20*2,20*2,4), np.uint8)
        cv2.imshow('image',img_tmp)
        k = cv2.waitKey(20) & 0xFF
        
        if k == ord('q'):
            return None
        if k == ord('r'):
            img = img_blank
        if k == ord('s'):
            cv2.imwrite(img_open+"_new.png", img)
            return img
        if k == ord('k'):
            cropRadius = cropRadius - 1
        if k == ord('l'):
            cropRadius = cropRadius + 1

if __name__ == '__main__':
    openImageForKeyring("output/testcar.png",False)