# ALPR

Automatic License Plate Recognition of brazilian license plates using 
Digital Image Processing and Machine Learning.

## Process

![Process](https://github.com/Drakmord2/alpr/blob/master/templates/process.png)

**Stages:**

- Enhancement **(1)**
  - Noise filtering
  - Homomorphic filtering
- Segmentation **(2)**
  - Thresholding
  - Morphologic transformation
- Character extraction **(3)**
  - Contours
- Character recognition **(4)**
  - K-Nearest Neighbors (k-NN)
