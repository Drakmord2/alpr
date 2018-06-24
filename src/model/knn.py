import numpy as np
import cv2 as cv
from util.csvUtil import CSVUtil


class KNN:
    def __init__(self):
        pass

    def train(self):
        pass

    def classify(self, candidates):
        print("  - KNN")
        result = ""

        classifications = np.loadtxt("../base/recognition/classifications.txt", np.float32)
        classifications = classifications.reshape((classifications.size, 1))
        flattened_images = np.loadtxt("../base/recognition/flattened_images.txt", np.float32)

        knn = cv.ml.KNearest_create()
        knn.train(flattened_images, cv.ml.ROW_SAMPLE, classifications)

        for file in candidates:
            img = cv.imread(file, 0)

            img_resized = cv.resize(img, (34, 52))
            img_resized = img_resized.reshape((1, 34 * 52))
            img_resized = np.float32(img_resized)

            retval, results, neigh_resp, dists = knn.findNearest(img_resized, k=1)

            current_char = str(chr(int(results[0][0])))

            result += current_char

        return result
    
    def get_training_data(self):
        digits = ["../templates/mandatory/boxes/z.png", "../templates/mandatory/boxes/y.png", "../templates/mandatory/boxes/x.png",
                  "../templates/mandatory/boxes/w.png", "../templates/mandatory/boxes/v.png", "../templates/mandatory/boxes/u.png",
                  "../templates/mandatory/boxes/t.png", "../templates/mandatory/boxes/s.png", "../templates/mandatory/boxes/r.png",
                  "../templates/mandatory/boxes/q.png", "../templates/mandatory/boxes/p.png", "../templates/mandatory/boxes/o.png",
                  "../templates/mandatory/boxes/n.png", "../templates/mandatory/boxes/m.png", "../templates/mandatory/boxes/l.png",
                  "../templates/mandatory/boxes/k.png", "../templates/mandatory/boxes/j.png", "../templates/mandatory/boxes/i.png",
                  "../templates/mandatory/boxes/h.png", "../templates/mandatory/boxes/g.png", "../templates/mandatory/boxes/f.png",
                  "../templates/mandatory/boxes/e.png", "../templates/mandatory/boxes/d.png", "../templates/mandatory/boxes/c.png",
                  "../templates/mandatory/boxes/b.png", "../templates/mandatory/boxes/a.png", "../templates/mandatory/boxes/0.png",
                  "../templates/mandatory/boxes/9.png", "../templates/mandatory/boxes/8.png", "../templates/mandatory/boxes/7.png", 
                  "../templates/mandatory/boxes/6.png", "../templates/mandatory/boxes/5.png", "../templates/mandatory/boxes/4.png", 
                  "../templates/mandatory/boxes/3.png", "../templates/mandatory/boxes/2.png", "../templates/mandatory/boxes/1.png"]

        classifications = [ord('Z'), ord('Y'), ord('X'), ord('W'), ord('V'), ord('U'), ord('T'), ord('S'), ord('R'),
                           ord('Q'), ord('P'), ord('O'), ord('N'), ord('M'), ord('L'), ord('K'), ord('J'), ord('I'),
                           ord('H'), ord('G'), ord('F'), ord('E'), ord('D'), ord('C'), ord('B'), ord('A'), ord('0'),
                           ord('9'), ord('8'), ord('7'), ord('6'), ord('5'), ord('4'), ord('3'), ord('2'), ord('1')]

        flattened_images = np.empty((0, 34 * 52))
        for digit in digits:
            img = cv.imread(digit, 0)
            img = cv.resize(img, (34, 52))

            flattened = img.reshape((1, 34 * 52))
            flattened_images = np.append(flattened_images, flattened, 0)

        classifications = np.array(classifications, np.float32)
        classifications = classifications.reshape(classifications.size, 1)

        np.savetxt("../base/classifications.txt", classifications)
        np.savetxt("../base/flattened_images.txt", flattened_images)
