import numpy as np
import cv2 as cv
from util.csvUtil import CSVUtil


class KNN:
    def __init__(self):
        pass

    def train(self):
        pass

    def classify(self, digits):
        print("  - KNN")
        result = ""

        npaClassifications = np.loadtxt("../bin/recognition/classifications.txt", np.float32)
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        npaFlattenedImages = np.loadtxt("../bin/recognition/flattened_images.txt", np.float32)

        kNearest = cv.ml.KNearest_create()

        kNearest.train(npaFlattenedImages, cv.ml.ROW_SAMPLE, npaClassifications)

        for file in digits:
            img = cv.imread(file, 0)

            imgROIResized = cv.resize(img, (34, 52))
            npaROIResized = imgROIResized.reshape((1, 34 * 52))
            npaROIResized = np.float32(npaROIResized)

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)

            strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

            result += strCurrentChar

        return result
    
    def get_data(self):
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

        np.savetxt("../bin/classifications.txt", classifications)
        np.savetxt("../bin/flattened_images.txt", flattened_images)
