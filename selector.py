import os
import cv2
import shutil

data = "data"
patches = "patches"
for file in os.listdir(os.path.join(data, patches)):
    image = cv2.imread(os.path.join(os.path.join(data, patches), file))
    image = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("image", image)
    key = cv2.waitKey(0)
    if key == 49:
        if not os.path.exists(os.path.join(data, patches + "_0.5")):
            os.makedirs(os.path.join(data, patches + "_0.5"))
        shutil.move(os.path.join(os.path.join(data, patches), file),
                    os.path.join(os.path.join(data, patches + "_0.5"), file))
    elif key == 50:
        if not os.path.exists(os.path.join(data, patches + "_0.33")):
            os.makedirs(os.path.join(data, patches + "_0.33"))
        shutil.move(os.path.join(os.path.join(data, patches), file),
                    os.path.join(os.path.join(data, patches + "_0.33"), file))
    elif key == 51:
        if not os.path.exists(os.path.join(data, patches + "_0.25")):
            os.makedirs(os.path.join(data, patches + "_0.25"))
        shutil.move(os.path.join(os.path.join(data, patches), file),
                    os.path.join(os.path.join(data, patches + "_0.25"), file))
    elif key == 52:
        if not os.path.exists(os.path.join(data, patches + "_uncertain")):
            os.makedirs(os.path.join(data, patches + "_uncertain"))
        shutil.move(os.path.join(os.path.join(data, patches), file),
                    os.path.join(os.path.join(data, patches + "_uncertain"), file))
    elif key == 53:
        if not os.path.exists(os.path.join(data, patches + "_normal")):
            os.makedirs(os.path.join(data, patches + "_normal"))
        shutil.move(os.path.join(os.path.join(data, patches), file),
                    os.path.join(os.path.join(data, patches + "_normal"), file))
    else:
        break
print("\nFinished converting the file!")