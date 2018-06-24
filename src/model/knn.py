import numpy as np
import cv2 as cv
from util.csvUtil import CSVUtil


# K-Nearest Neighbors
class KNN:
    def __init__(self):
        self.knn = self.train()

    def train(self):
        labels = np.loadtxt("../base/recognition/labels.txt", np.float32)
        labels = labels.reshape((labels.size, 1))
        flattened_images = np.loadtxt("../base/recognition/flattened_images.txt", np.float32)

        knn = cv.ml.KNearest_create()
        knn.train(flattened_images, cv.ml.ROW_SAMPLE, labels)

        return knn

    def classify(self, candidates):
        print("  - KNN")

        result = ""

        for file in candidates:
            img = cv.imread(file, 0)

            img_resized = cv.resize(img, (34, 52))
            img_resized = img_resized.reshape((1, 34 * 52))
            img_resized = np.float32(img_resized)

            retval, results, neigh_resp, dists = self.knn.findNearest(img_resized, k=1)

            current_char = str(chr(int(results[0][0])))

            result += current_char

        return result
    
    def get_training_data(self):
        image_dir = "../templates/mandatory/boxes"

        digits = [image_dir+"/z.png", image_dir+"/y.png", image_dir+"/x.png",
                  image_dir+"/w.png", image_dir+"/v.png", image_dir+"/u.png",
                  image_dir+"/t.png", image_dir+"/s.png", image_dir+"/r.png",
                  image_dir+"/q.png", image_dir+"/p.png", image_dir+"/o.png",
                  image_dir+"/n.png", image_dir+"/m.png", image_dir+"/l.png",
                  image_dir+"/k.png", image_dir+"/j.png", image_dir+"/i.png",
                  image_dir+"/h.png", image_dir+"/g.png", image_dir+"/f.png",
                  image_dir+"/e.png", image_dir+"/d.png", image_dir+"/c.png",
                  image_dir+"/b.png", image_dir+"/a.png", image_dir+"/0.png",
                  image_dir+"/9.png", image_dir+"/8.png", image_dir+"/7.png",
                  image_dir+"/6.png", image_dir+"/5.png", image_dir+"/4.png",
                  image_dir+"/3.png", image_dir+"/2.png", image_dir+"/1.png"]

        labels = [ord('Z'), ord('Y'), ord('X'), ord('W'), ord('V'), ord('U'), ord('T'), ord('S'), ord('R'),
                  ord('Q'), ord('P'), ord('O'), ord('N'), ord('M'), ord('L'), ord('K'), ord('J'), ord('I'),
                  ord('H'), ord('G'), ord('F'), ord('E'), ord('D'), ord('C'), ord('B'), ord('A'), ord('0'),
                  ord('9'), ord('8'), ord('7'), ord('6'), ord('5'), ord('4'), ord('3'), ord('2'), ord('1')]

        flattened_images = np.empty((0, 34 * 52))
        for digit in digits:
            img = cv.imread(digit, 0)
            img = cv.resize(img, (34, 52))

            flattened = img.reshape((1, 34 * 52))
            flattened_images = np.append(flattened_images, flattened, 0)

        labels = np.array(labels, np.float32)
        labels = labels.reshape(labels.size, 1)

        np.savetxt("../base/recognition/labels.txt", labels)
        np.savetxt("../base/recognition/flattened_images.txt", flattened_images)
