import cv2
import numpy as np
from PIL import Image
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw
import os

class fragile(object):
    class Break(Exception):
      """Break out of the with statement"""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value.__enter__()

    def __exit__(self, etype, value, traceback):
        error = self.value.__exit__(etype, value, traceback)
        if etype == self.Break:
            return True
        return error

class draw:

    def text_image(text_path, font_path=None, font_size=45):
        """
        Convert .txt file to image
        input: text_path (path to .txt file)
               font_path (path to font file; default=FreeMono.ttf builtin font)
               font_size (ASCII font size in image; default=45)
        return: Pillow image
        """
        
        PIXEL_ON = 0 
        PIXEL_OFF = 255
        grayscale = 'L'
        
        with open(text_path) as text_file:
            lines = tuple(l.rstrip() for l in text_file.readlines())
 
        try:
            font = PIL.ImageFont.truetype(font_path, size=font_size)
        except:
            font = PIL.ImageFont.truetype("FreeMono.ttf", size=font_size, layout_engine=PIL.ImageFont.LAYOUT_RAQM) 
 

        pt2px = lambda pt: int(round(pt * 96.0 / 72)) 
        max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
        
        test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        max_height = pt2px(font.getsize(test_string)[1])
        max_width = pt2px(font.getsize(max_width_line)[0])
        height = max_height * len(lines)
        width = int(round(max_width + 40))
        image = PIL.Image.new(grayscale, (width, height), color=PIXEL_OFF)
        draw = PIL.ImageDraw.Draw(image)
        
        vertical_position = 5
        horizontal_position = 5
        line_spacing = int(round(max_height * 0.8))
        for line in lines:
            draw.text((horizontal_position, vertical_position),
                      line, fill=PIXEL_ON, font=font)
            vertical_position += line_spacing
        
        c_box = PIL.ImageOps.invert(image).getbbox()
        image = image.crop(c_box)
        
        return image

    def scale_image(image, new_width=100):
        """
        Resizes an image preserving the aspect ratio
        input: image (Pillow image)
               new_width (Scale image smaller for ease; default=100)
        return: Pillow image
        """
        (original_width, original_height) = image.size
        aspect_ratio = original_height/float(original_width)
        new_height = int(aspect_ratio * new_width)

        new_image = image.resize((new_width, new_height))
        return new_image

    @staticmethod
    def mk_ascii(image, ASCII=["A","B","C","D","E","F","I","J","K","N","P","R","S","V","Y","2","3","4","5","6","7","8","9"], fontpath=None, fontsize=45):
        """
        Turn image into colorized ASCII art
        input: image (3D numpy array uint8)
               ASCII (ASCII list of string chars to build image; default=longlist)
               font_path (path to font file; default=FreeMono.ttf builtin font)
               font_size (ASCII font size in image; default=45)
        return: 3D numpy array uint8
        """
        
        lower = np.array([0, 0, 0])
        upper = np.array([254,254,254])

        gray = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        gray = draw.scale_image(gray)
        gray = gray.convert("L")
        width, height = gray.size

        pixels = gray.getdata()
        try:
            characters = "".join([ASCII[pixel//len(ASCII)] for pixel in pixels])
        except:
            ASCII=["A","B","C","D","E","F","I","J","K","N","P","R","S","V","Y","2","3","4","5","6","7","8","9"]
            characters = "".join([ASCII[pixel//len(ASCII)] for pixel in pixels])
            
        pix_count = len(characters)
        ascii_image = "\n".join([characters[i:(i+width)] for i in range(0, pix_count,width)]) 
            
        with fragile(open("ascii_image.txt", "w")) as f:
            f.write(ascii_image)
            f.close()
            raise fragile.Break

        im = draw.text_image('ascii_image.txt', fontpath, fontsize)
        gray = cv2.resize(np.asarray(im), (image.shape[1], image.shape[0]))

        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        mask = cv2.inRange(gray, lower, upper)

        gray[mask!=0] = image[mask!=0]
        gray[mask==0] = [0,0,0]

        return gray
        

        

