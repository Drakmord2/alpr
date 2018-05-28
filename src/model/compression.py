import numpy as np
from matplotlib import pyplot as plt


class Compression:

    def __init__(self, image):
        self.image = image

    def huffman(self):
        print('  - Huffman Coding')

        prob = self.get_probability()
        root = self.huffman_tree(prob)

        self.statistics(prob, root)

        encoded = self.encode(root)

        return encoded, root

    def huffman_tree(self, dict):
        list = sorted(dict.items(), key=lambda d: d[1])
        nodes = []
        for pixel in list:
            nodes.append(Node(pixel[0], pixel[1]))

        while len(nodes) > 1:
            list = sorted(nodes, key=lambda n: n.prob)

            left = list[0]
            right = list[1]
            prob_l = left.prob
            prob_r = right.prob

            prob_node = prob_l + prob_r
            node = Node(None, prob_node)
            node.set_child(left, right)

            nodes.remove(left)
            nodes.remove(right)

            nodes.append(node)

        root = nodes[0]

        return root

    def get_code(self, s, node, code):
        if node.value is None:
            self.get_code(s + "0", node.left, code)
            self.get_code(s + "1", node.right, code)
        else:
            if not s:
                code[node.value] = "0"
            else:
                code[node.value] = s

    def encode(self, root):
        print('    - Encoding')
        code = {}
        self.get_code("", root, code)

        x, y = self.image.shape
        encoded = [['' for j in range(y)] for i in range(x)]
        # encoded = np.zeros((x, y))

        for i in range(x):
            for j in range(y):
                value = self.image[i][j]
                encoded[i][j] = code[value]

        return encoded

    def decode_huffman(self, encoded, root):
        print('  - Huffman Decoding')

        code = {}
        self.get_code("", root, code)

        x = len(encoded)
        y = len(encoded[0])
        # x, y = encoded.shape
        decoded = np.zeros((x, y))

        for i in range(x):
            for j in range(y):
                code_val = encoded[i][j]
                value = 0
                for k, v in code.items():
                    if code_val == v:
                        value = k
                        break

                decoded[i][j] = value

        return decoded

    def get_probability(self):
        dict = {}
        x, y = self.image.shape

        for p in range(256):
            dict[p] = 0

        for i in range(x):
            for j in range(y):
                value = self.image[i][j]
                dict[value] = dict[value] + 1

        dict = {k: v for k, v in dict.items() if v != 0}

        distribution = {}
        total_pixels = x * y
        for k, v in dict.items():
            prob = v / total_pixels
            distribution[k] = prob

        return distribution

    def statistics(self, prob, root):
        print('    - Statistics: ')
        code = {}
        self.get_code("", root, code)

        l_avg = sum([len(v) * prob[k] for k, v in code.items()])
        entropy = sum([-i[1] * np.log2(i[1]) for i in prob.items()])
        print('        - Entropy: ', round(entropy, 3), ' | Lavg: ', round(l_avg, 3))

        x, y = self.image.shape
        total = x * y
        n1 = total * 8
        n2 = sum([len(v) * prob[k] * total for k, v in code.items()])
        cr = n1 / n2
        rd = 1 - 1 / cr

        print('        - Original Size: ', self.scale(n1), ' | Compressed Size: ', self.scale(n2))
        print('        - Compression rate: ', round(cr, 3), ' | Redundancy: ', round(rd, 3))

    def scale(self, bits):
        if bits < 1024:
            return str(bits) + "b"

        kb = round(bits / 1024, 3)
        value = str(kb) + "Kb"

        if kb > 1024:
            mb = round(kb / 1024, 3)
            value = str(mb) + "Mb"

        return value

    def run_length(self, data):
        print('  - Run-Lengh Encoding')
        lines = len(data)
        col = len(data[0])
        rle = ''

        for i in range(lines):
            run = ''
            counter = 0
            for j in range(col-1):
                counter += 1

                if data[i][j] != data[i][j+1] or j == col-2:
                    if j == col-2:
                        counter += 1

                    if data[i][j] == '0':
                        run += str(counter) + "W"
                        counter = 0
                        continue

                    run += str(counter) + "B"
                    counter = 0

            rle += run + '\n'

        return rle

    def decode_rle(self):
        pass


class Node(object):
    value = None
    prob = 0
    left = None
    right = None

    def __init__(self, value, prob):
        self.value = value
        self.prob = prob

    def set_child(self, left, right):
        self.left = left
        self.right = right

    def __cmp__(self, other):
        return self.prob > other.prob

    def __repr__(self):
        return "Pixel_value -> %s | Prob -> %s | (%s !!! %s)" % (self.value, self.prob, self.left, self.right)
