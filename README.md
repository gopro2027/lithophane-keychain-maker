# lithophane-keychain-maker
create lithophane keychains from cars in python

Program for turning pictures of photos into keychains.
2 types: raw car & medallion (circle around it)

Works by removing the background of the image, then setting the background pixels in pure black or white to set the height/thickness before turning it into a lithophane.
For example, the keyring is added by adding a black circle to the image.
The background is removed through the removebg website (AI based). It's a little sneaky though, because it cost money for full res images and i don't want to pay, the code just gets the free low res example images and upscales the result to the full res image and copies the background removed. Quite an overseight on their end I'd say, but there's still a max limit of how many you can do. It looses a slight bit of precision but it looks perfectly find especially since it's 3d printed and that looses a lot of percision due to printing limitations.

Credits:
Some random image to lithophane python project... idk really how it works I just messed with it until it did what I wanted.
Removebg website/cli
