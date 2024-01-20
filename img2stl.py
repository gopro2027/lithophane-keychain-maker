# https://github.com/colbrydi/Lithophane/blob/master/Lithophane_Example.ipynb

# TODO: Make an intermediate program to allow me to make a hole of opacity 
# or a ring of solid black on the image at a specified location to make 
# the spot to put the keychain on. Can use width to calculate size for an X-mm hole

#user params to edit
#width = 75 #Width in mm
perimetercalc = 105 #half of total perimeter
thickness = 1.8 #thickness in mm    1.8 only looks good with a light directly behind it               
base = 0.4 #base thickness in mm    0.3 was the smallest that game good results on the back side of the print

# https://www.remove.bg/ to remove backgrounds with extra pizazz
# www.slazzer.com does not have the shadow under it but is the same as removebg

base_dir = "raw_images/"
imginput = 'bluec5above.jpg'
#imginput = 'drewbike-removebg.png'




import os
import time
import numpy as np
from stl import mesh
from pathlib import Path
from PIL import Image, ImageFilter
import cv2
#from rembg import remove
import lithophane_colbrydi as li
import addkeyring as ak

#def remove_bg(image):
#    return remove(image)  #rembg pip

def getAPIKey():
    return "XXXXXXXXXXXXXXXXXXXXXXXX" # PUT YOUR API KEY FROM REMOVE.BG HERE

# https://github.com/remove-bg/remove-bg-cli/releases/
# If there is an error about bootstrap.js, delete this folder:
#    C:\Users\Tyler\AppData\Local\Temp\pkg\b544b96dfa4cb284cbc1d94eac17dc8dc964f5b84a011214227bae0ff6f25dd2
def remove_bg(image_filepath):
    imagenobg = "output/rembg/"+Path(image_filepath).stem+".png"
    outputimg = "output/"+Path(image_filepath).stem+".png"

    #only call api if image has not been created
    #consider updating to pip install rembg https://github.com/danielgatis/rembg
    if not os.path.exists(imagenobg):
        # https://github.com/remove-bg/remove-bg-cli
        cmd = "removebg --api-key "+getAPIKey()+" "+image_filepath+" --format png --size preview --type car --output-directory output/rembg"
        os.system(cmd)

    imgorig = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
    imgorig = cv2.cvtColor(imgorig, cv2.COLOR_RGB2RGBA)
    height, width, channels = imgorig.shape
    img = cv2.imread(imagenobg, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    # run smoothing algorithm on image returned so that it 
    img = cv2.medianBlur(img,13)

    #ret, mask = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    #imgorig[:, :, 3] = img[:,:,3] #copy the alpha layer over to the new image
    

    #alternative way to copy the alpha layer, but also setting values in the background to black
    for y in range(0, height):
        for x in range(0, width):
            a = img[y, x, 3]
            if a < 255:
                imgorig[y,x] = [0,0,0,0]



    #dst = cv2.bitwise_and(imgorig, imgorig, mask=mask)  OLD

    #b, g, r = cv2.split(imgorig)
    #rgba = [b,g,r, mask]
    #dst = cv2.merge(rgba,4)

    cv2.imwrite(outputimg,imgorig)

    #while(1):
    #    cv2.imshow('image',dst)
    #    k = cv2.waitKey(20) & 0xFF
    #    if k == ord('q'):
    #        break
    

    return outputimg


def smooth_img(filename):
    image = Image.open(filename)
    image = image.filter(ImageFilter.ModeFilter(size=13))
    image.save(filename)
    #vesselImage = cv2.imread(filename) #the original image
    #cv2.threshold(vesselImage, vesselImage, 125, 255, cv2.THRESH_BINARY);
    #blurredImage = cv2.pyrUp(vesselImage, blurredImage);

    #for i in range(0,15):
    #    cv2.medianBlur(blurredImage, blurredImage, 7);

    #cv2.pyrDown(blurredImage, blurredImage);
    #cv2.threshold(blurredImage, blurredImage, 200, 255, cv2.THRESH_BINARY);

def crop_bg(image):
    return image.crop(image.getbbox())

def open(image_str):
    return Image.open(image_str)

def save(image, output):
    image.save(output)

def calcWidthFromMaxSize(image, maxSize):
    height = image.height
    width = image.width
    ratio = float(width)/float(height)
    for i in range(1,maxSize):
        tot = i + i * ratio
        if tot > maxSize:
            return int((i - 1) * ratio) #width
    print("Error calculating size!")
    return maxSize
    #if width > height:
    #    return maxSize
    #else:
    #    return int(float(width) / (float(height)/float(maxSize)))

def rotateSTL(file, deg):
    stl_data = mesh.Mesh.from_file(file)
    #c = np.cos(np.deg2rad(deg))
    #s = np.sin(np.deg2rad(deg))
    #matrix = np.array([[1, 0, 0],
    #               [0, c, -s],
    #               [0, s,c]])

    #R[:3, :3] = matrix
    #print(R)
    #stl_data.transform(matrix)
    stl_data.rotate([1, 0, 0], np.deg2rad(deg))

    stl_data.save(file)

#unused
def transparent_to_white(input,output):
    image = Image.open(input).convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)

    new_image.convert("RGB").save(output)




#process the image
filename = remove_bg(base_dir+imginput)
image = open(filename)
#if removeBG == True:
#    image = remove_bg(image) #might want to not run this if it gets bad results, and use www.remove.bg
image = crop_bg(image)

#save the processed image to the output folder
#filename="output/"+imginput[:-4] + '.png'
save(image,filename)

width = calcWidthFromMaxSize(image, perimetercalc)
print("Calculated width = "+str(width))

image = ak.openImageForKeyring(filename,width,True)
cv2.imwrite(filename, image)

#filename = "output/testcar.png"
#convert the image to a lithophane
x,y,z = li.jpg2stl(filename, width=width, h = thickness, d = base, show=False)


#save to stl
model = li.makemesh(x,y,z)
filename="output/"+imginput[:-4] + '.stl'
model.save(filename)

rotateSTL(filename,90)
