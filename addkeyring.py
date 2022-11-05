import cv2
import numpy as np
import math

holeSize = 2.5 #diameter
holeWallSize = 6 #diameter


#holeWallSize = holeWallSize/2 - holeSize #div by 2 bc 2 sides and subtract the hole size
#holeSize = holeSize / 2           #convert to radius for later calculation

holeWallSize = holeWallSize / 2 #div by 2 to get radius
holeSize = holeSize / 2           #convert to radius for later calculation

mouseX = 0
mouseY = 0
img_width = 0
img_height = 0

def eraseCircle(img,x,y,radius, color, avoidNonAlpha):
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
                if a < 0.95 or avoidNonAlpha == False: #spot is transparent
                    img[y + iy, x + ix] = color
            except:
                #print("Error!")
                fdya = 0 #literally nothing
    #exit(0)
    return img


def create_keyring(img, x,y, tmpRing):
    global holeSizePixel, holeWallSizePixel,img_width,img_height
    #img_write = img.copy() #will draw on top of this one
    #cv2.circle(img,(x,y),int(holeSizePixel),(255,0,0,255),int(holeWallSizePixel),2) #ring
    #cv2.circle(img,(x,y),int(holeSizePixel),(255,0,0,255),int(holeWallSizePixel),2) #hole
    #cv2.circle(img_write,(x,y),int(holeSizePixel),(0,0,0,0),-1) #center

    eraseCircle(img,x,y,holeWallSizePixel,[255,0,0,255] if tmpRing else [0,0,0,255],True)
    eraseCircle(img,x,y,holeSizePixel,[0,0,0,0],False)

    #img = cv2.addWeighted(img, 1, img_write, 1, 0)
    return img #eraseCircle(img,x,y,int(holeSizePixel))

def mouse_event(event,x,y,flags,param):
    global mouseX, mouseY, img, hasSelectedHole
    mouseX = x
    mouseY = y
    if event != 0:
        print("event: "+str(event))

    if event == 1: # 1 is the actual click button based on output    cv2.EVENT_LBUTTONDBLCLK is 7:
        img = create_keyring(img, x,y, False)
        hasSelectedHole = True

def openImageForKeyring(img_open,widthOfImageInMM,exitOnSelect):
    global img, img_orig, img_width_mm, mouseX, mouseY, holeSizePixel, holeWallSizePixel,img_width,img_height, hasSelectedHole
    hasSelectedHole = False
    img_width_mm = widthOfImageInMM
    img_orig = cv2.imread(img_open, cv2.IMREAD_UNCHANGED)
    img = img_orig.copy()
    height, width, channels = img.shape
    img_width = width
    img_height = height
    holeSizePixel = holeSize * (width/widthOfImageInMM)
    holeWallSizePixel = holeWallSize * (width/widthOfImageInMM)

    print("press q to cancel image editing")
    print("press r to remove keyring")
    print("press s to save current image")
    print("left click to place keyring")
    #cv2.resizeWindow('image', width, height)
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image',mouse_event)
    while(1):
        if hasSelectedHole == True and exitOnSelect == True:
            return img
        img_tmp = img.copy()
        img_tmp = create_keyring(img_tmp,mouseX,mouseY, True)
        cv2.imshow('image',img_tmp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            return None
        if k == ord('r'):
            img = img_orig
        if k == ord('s'):
            cv2.imwrite(img_open+"_new.png", img)
            return img

if __name__ == '__main__':
    openImageForKeyring("output/testcar.png",75,False)