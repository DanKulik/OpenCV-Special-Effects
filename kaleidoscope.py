import cv2
import numpy as np
from numba import njit
from PIL import Image, ImageDraw


class draw:

    @staticmethod
    @njit(cache=True, fastmath=True)
    def to_squares(inp):
        """
        Map image coordinates to square geometry
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        result = np.zeros(inp.shape)

        ll = len(inp) 

        for x, row in enumerate(inp):
            unit_x = x / ll * 2 - 1
            rr = len(row)

            for y, _ in enumerate(row):
                unit_y = y / rr * 2 - 1

                x2 = unit_x * unit_x
                y2 = unit_y * unit_y
                r2 = x2 + y2
                rad = np.sqrt(r2 - x2 * y2)

                if r2 < 1e-5:
                    continue

                reciprocal_sqrt = 1.0 / np.sqrt(r2)

                u = unit_x * rad * reciprocal_sqrt
                v = unit_y * rad * reciprocal_sqrt

                u = (u + 1) / 2 * ll 
                v = (v + 1) / 2 * rr

                result[x][y] = inp[round(u)][round(v)]
                
        return result

    @staticmethod
    @njit(cache=True, fastmath=True)
    def to_circles(inp):
        """
        Map image coordinates to circular geometry
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        result = np.zeros_like(inp)

        ll = len(inp)

        for x, row in enumerate(inp):
            unit_x = x / ll * 2 - 1
            rr = len(row)

            for y, _ in enumerate(row):
                unit_y = y / rr * 2 - 1

                u2 = unit_x * unit_x
                v2 = unit_y * unit_y
                r2 = u2 + v2

                if r2 > 1:
                    continue

                uv = unit_x * unit_y
                fouru2v2 = 4.0 * uv * uv
                rad = r2 * (r2 - fouru2v2)
                
                if uv == 0.0:
                    sgnuv = 0.0
                elif uv < 0:
                    sgnuv = -1.0
                else:
                    sgnuv = 1.0
                    
                sqrto = np.sqrt(0.5 * (r2 - np.sqrt(rad)))

                if abs(unit_x) > 1e-5:
                    v = sgnuv / unit_x * sqrto
                else:
                    v = unit_y

                if abs(unit_y) > 1e-5:
                    u = sgnuv / unit_y * sqrto
                else:
                    u = unit_x

                u = (u + 1) / 2 * ll 
                v = (v + 1) / 2 * rr

                result[x][y] = inp[round(u)][round(v)]

        return result

    def rotate(img, angle):
        """
        Rotate image
        input: img (Pillow image)
        return: (Pillow image)
        """
        
        img = img.rotate(angle)
        return img

    def flipH(img):
        """
        Horizontally flip image
        input: img (Pillow image)
        return: (Pillow image)
        """
        
        data = (img.width, 0, 0, img.height)
        img = img.transform((img.width,img.height), Image.EXTENT, data)
        return img

    def flipV(img):
        """
        Vertically flip image
        input: img (Pillow image)
        return: (Pillow image)
        """
        
        data = (0, img.height, img.width, 0)
        img = img.transform((img.width,img.height), Image.EXTENT, data)
        return img

    def flipHV(img):
        """
        Horizontally and vertically flip image
        input: img (Pillow image)
        return: (Pillow image)
        """

        img = draw.flipV(img)
        img = draw.flipH(img)
        return img

    def thumbnail(img, width, height, mode):
        """
        Thumbnail image
        input: img (Pillow image)
               width (image width)
               height (image height)
               mode (Pillow interpolation mode eg: Image.NEAREST)
        return: (Pillow image)
        """
        
        img.thumbnail((width,height), mode)
        width, height = img.size
        return img

    def combine(iimg0, iimg1, iimg2, iimg3):
        """
        Combines multiple images 
        input: iimg0, iimg1, iimg2, iimg3 (Pillow images)
        return: (Pillow image)
        """
        
        width, height = iimg0.width, iimg0.height
        newSize = (width*2, height*2)
        canvas = Image.new('RGB', newSize, 'black')
        canvas.paste(iimg0, (0, 0))
        canvas.paste(iimg1, (width, 0))
        canvas.paste(iimg2, (0, height))
        canvas.paste(iimg3, (width, height))
        canvas = draw.thumbnail(canvas, width, height, Image.NEAREST)
        return canvas

    def kal(img):
        """
        Combines multiple image flips 
        input: img (3D numpy array uint8)
        return: (Pillow image)
        """
        
        img = Image.fromarray(np.uint8(img)).convert('RGB')
            
        img0 = draw.flipH(img.copy())
        img1 = draw.flipV(img.copy())
        img2 = draw.flipHV(img.copy())
        return draw.combine(img, img0, img1, img2)
        
    @staticmethod
    def kaleidoscope(img, resize=True):
        """
        Creates a kaleidoscope image 
        input: img (3D numpy array uint8)
               resize (Bool: True returns image with new size
                             False returns image with original size;
                             default=True
        return: (3D numpy array uint8)
        """

        frame = draw.kal(img) 

        frame = cv2.resize(np.asarray(frame),(1000,1000))
        frame = draw.to_circles(frame)
        frame = Image.fromarray(np.uint8(frame)).convert('RGB')
        
        frame1 = np.asarray(draw.rotate(frame, 0))
        frame2 = np.asarray(draw.rotate(frame, 90))
        frame3 = np.asarray(draw.rotate(frame, -45))
        frame4 = np.asarray(draw.rotate(frame, 45))

        frame = np.uint8(0.25*frame1 + 0.25*frame2 + 0.25*frame3 + 0.25*frame4)

        frame = draw.to_squares(frame)
        h, w = frame.shape[:2]
        h, w = int(h/2) , int(w/2)
        frame[h-1:h+2,w-1:w+2] = frame[h-8,w-8]

        if not resize:
            frame = cv2.resize(frame,(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        return frame

         

        

