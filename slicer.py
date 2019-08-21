import os
import sys
import cv2
import time
import gdal
import numpy as np

def judge(gray, row, col, size):
    row_lower = row - size
    row_higher = row + size
    col_lower = col - size
    col_higher = col + size
    if row_lower < 0:
        row_lower = 0
    if col_lower < 0:
        col_lower = 0
    if row_higher > 2999:
        row_higher = 2999
    if col_higher > 2999:
        col_higher = 2999
    if np.sum(gray[row_lower:row_higher, col_lower:col_higher] < 170) > 960:
        return True
    else:
        return False


data = "data"
tif = "tif"
patches = "patches"
size = 3000
win = 1000
gdal.AllRegister()
num = 0
num1 = 0
num2 = 0
start_time = time.time()
for image in os.listdir(os.path.join(data, tif)):
    tif = gdal.Open(os.path.join(os.path.join(data, tif), image))
    width = size
    height = size
    width_ = tif.RasterXSize
    height_ = tif.RasterYSize
    sum = (int((width_-width)/win)+1) * (int((height_-height)/win)+1)
    print("This file will be cut into %d patches" % sum)
    while width < width_:
        while height < height_:
            newData = np.zeros([size, size, 3])
            band = tif.GetRasterBand(1)
            r = band.ReadAsArray(width - size, height - size, size, size)
            band = tif.GetRasterBand(2)
            g = band.ReadAsArray(width - size, height - size, size, size)
            band = tif.GetRasterBand(3)
            b = band.ReadAsArray(width - size, height - size, size, size)
            newData[:, :, 0] = r
            newData[:, :, 1] = g
            newData[:, :, 2] = b
            image = cv2.merge([b, g, r])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            i = 0
            for row in range(0, size, 20):
                for col in range(0, size, 20):
                    if judge(gray, row, col, 40):
                        i = i+1
            if i < 800:
                num2 += 1
                height += win
                sys.stdout.write('\r>> Converting image %d/%d' % (num + num1 + num2, sum))
                sys.stdout.flush()
                continue
            if not os.path.exists(os.path.join(data, patches)):
                os.makedirs(os.path.join(data, patches))
            if i > 14000:
                cv2.imwrite(os.path.join(os.path.join(data, patches), str(num) + "_" + str(i) + ".jpg"), image)
                num += 1
            else:
                num1 += 1
            sys.stdout.write('\r>> Converting image %d/%d' % (num + num1 + num2, sum))
            sys.stdout.flush()
            height += win
        height = size
        width += win
    end_time = time.time()
    duration = end_time - start_time
    print("\nFinished converting the file!")
    print("duration:", duration)
print("Finished converting all the files!")

# file1:duration: 314.7803177833557
# file2:duration: 2715.0602118968964
# file3:duration: 380.25801372528076
# file4:duration: 458.0585947036743
# file5:duration: 204.41061091423035
# file6:duration: 442.2315082550049
# file7:duration: 1333.7072854042053
# file8:duration: 195.03005528450012
