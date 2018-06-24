import csv


class CSVUtil:
    def __init__(self):
        pass

    def write_moment(self, hulogs):
        with open('../bin/moments.csv', 'w', newline='') as csvfile:
            fieldnames = ['digit', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            digits = ["Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I",
                      "H", "G", "F", "E", "D", "C", "B", "A", "0", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
            for i in range(len(hulogs)):
                moment = hulogs[i]

                writer.writerow({
                    'digit': digits[i],
                    'u1': moment[0],
                    'u2': moment[1],
                    'u3': moment[2],
                    'u4': moment[3],
                    'u5': moment[4],
                    'u6': moment[5],
                    'u7': moment[6]})

    def read(self, path):
        data = []

        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                data.append(row)

        return data
