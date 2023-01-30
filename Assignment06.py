import numpy as np
import cv2
import os

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1]:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = 126
        return img;

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 2] = 255
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 126
        return img;

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0] // 3, self.position[1]:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 3:self.position[0] + 2 * self.size[0] // 3, self.position[1]:self.position[1] + self.size[1]] = 126
        img[self.position[0]+ 2 * self.size[0] // 3:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = 255
        return img;

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 3] = 255
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + self.size[1] // 3:self.position[1] + 2 * self.size[1] // 3] = 126
        img[self.position[0]:self.position[0] + self.size[0], self.position[1] + 2 * self.size[1] // 3:self.position[1] + self.size[1]] = 255
        return img;

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype = np.uint8)
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1]:self.position[1] + self.size[1] // 2] = 126
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1] // 2] = 255
        img[self.position[0]:self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 255
        img[self.position[0]+ self.size[0] // 2:self.position[0] + self.size[0], self.position[1] + self.size[1] // 2:self.position[1] + self.size[1]] = 126
        
        return img;

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        #print(self.size)

        cv2.imshow("Haar Feature", X)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum or subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        To see what the different feature types look like, check the Haar Feature Types image

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        y, x = self.size
        topLy, topLx = self.position[0] - 1, self.position[1] - 1
        score = 0

        if self.feat_type == (2, 1):
            topLeft = (topLy, topLx)
            topRight = (topLy, topLx + x)
            midLeft = (topLy + y//2, topLx)
            midRight = (topLy + y//2, topLx + x)
            botLeft = (topLy + y, topLx)
            botRight = (topLy + y, topLx + x)

            white = ii[midRight] + ii[topLeft] - ii[midLeft] - ii[topRight]
            gray = ii[botRight] + ii[midLeft] - ii[botLeft] - ii[midRight]
            score = white - gray
            return score

        if self.feat_type == (1, 2):
            topLeft = (topLy, topLx)
            topMid = (topLy, topLx + x//2)
            topRight = (topLy, topLx + x)
            botLeft = (topLy + y, topLx)
            botMid = (topLy + y, topLx + x//2)
            botRight = (topLy + y, topLx + x)

            white = ii[botMid] - ii[botLeft] - ii[topMid] + ii[topLeft]
            gray = ii[botRight] - ii[botMid] - ii[topRight] + ii[topMid]
            score = white - gray
            return score

        if self.feat_type == (3, 1):
            topLeft = (topLy, topLx)
            topRight = (topLy, topLx + x)
            mid1Left = (topLy + y//3, topLx)
            mid1Right = (topLy + y//3, topLx + x)
            mid2Left = (topLy + (2 * y)//3, topLx)
            mid2Right = (topLy + (2 * y)//3, topLx + x)
            botLeft = (topLy + y, topLx)
            botRight = (topLy + y, topLx + x)

            white1 = ii[mid1Right] - ii[mid1Left] - ii[topRight] + ii[topLeft]
            white2 = ii[botRight] - ii[botLeft] - ii[mid2Right] + ii[mid2Left]
            gray = ii[mid2Right] - ii[mid2Left] - ii[mid1Right] + ii[mid1Left]
            score = (white1 + white2) - gray
            return score

        if self.feat_type == (1, 3):
            topLeft = (topLy, topLx)
            topMid1 = (topLy, topLx + x//3)
            topMid2 = (topLy, topLx + (2 * x)//3)
            topRight = (topLy, topLx + x)
            botLeft = (topLy + y, topLx)
            botMid1 = (topLy + y, topLx + x//3)
            botMid2 = (topLy + y, topLx + (2 * x)//3)
            botRight = (topLy + y, topLx + x)

            white1 = ii[botMid1] - ii[botLeft] - ii[topMid1] + ii[topLeft]
            white2 = ii[botRight] - ii[botMid2] - ii[topRight] + ii[topMid2]
            gray = ii[botMid2] - ii[botMid1] - ii[topMid2] + ii[topMid1]
            score = (white1 + white2) - gray
            return score

        if self.feat_type == (2, 2):
            topLeft = (topLy, topLx)
            topMid = (topLy, topLx + x//2)
            topRight = (topLy, topLx + x)
            midLeft = (topLy + y//2, topLx)
            midMid = (topLy + y//2, topLx + x//2)
            midRight = (topLy + y//2, topLx + x)
            botLeft = (topLy + y, topLx)
            botMid = (topLy + y, topLx + x//2)
            botRight = (topLy + y, topLx + x)

            white1 = ii[midRight] - ii[midMid] - ii[topRight] + ii[topMid]
            white2 = ii[botMid] - ii[botLeft] - ii[midMid] + ii[midLeft]
            gray1 = ii[midMid] - ii[midLeft] - ii[topMid] + ii[topLeft]
            gray2 = ii[botRight] - ii[botMid] - ii[midRight] + ii[midMid]
            score = (white1 + white2) - (gray1 + gray2)
            return score

def convert_image_to_integral_image(img):
    """Convert a list of grayscale images to integral images.

    Args:
        image :  Grayscale image (uint8 or float).

    Returns:
        2d Array : integral image.
    """
    integral = np.zeros_like(img).astype(int)
    # print(img)
    numRows = img.shape[0]
    numCols = img.shape[1]
    for r in range(0, numRows):
        for c in range(0, numCols):
            if r > 0 and c > 0:
                integral[r, c] = img[r, c] + integral[r-1, c] + integral[r, c-1] - integral[r-1, c-1]
            elif r > 0:
                integral[r, c] = img[r, c] + integral[r-1, c]
            elif c > 0:
                integral[r, c] = img[r, c] + integral[r, c - 1]
            else:
                integral[r, c] = img[r, c]
    return integral

#for this class, you will write the train, predict and faceDetect functions
class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        #now create all possible haar features that would work in a 24x24 window
        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(2*feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(2*feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights = np.hstack((np.full(len(self.negImages), np.reciprocal(2 * len(self.negImages), dtype=np.float32)),\
                np.full(len(self.posImages), np.reciprocal(2 * len(self.posImages), dtype=np.float32))))

        bin = np.vectorize(ViolaJones._bin)
        for t in range(num_classifiers):
            weights /= np.sum(weights)
            h = VJ_Classifier(scores, self.labels, weights)
            h.train(self.haarFeatures)
            self.classifiers.append(h)

            predictions = [h.predict(ii) for ii in self.integralImages]
            r = h.error / (1 - h.error)
            weights *= np.power(r, (1 - bin(predictions)))
            self.alphas.append(np.log(1/r)) 

    def _bin(p):
        if p == 1:
            return 0
        else:
            return 1

    def predictHelper(self, image):
        predictions = [a.predict(image) for a in self.classifiers]
        finalSum = np.sum(self.alphas * np.array(predictions))
        alphaSum = np.sum(self.alphas)
        if finalSum >= alphaSum * 0.5:
            return 1
        else:
            return -1

    def predict(self, image):
        """Return prediction for a given image.
        Use the strong classifier you've created to determine if a given image
        is a face or not.  Looking at the algorithm, you will need alphas, which you
        should have saved in train, and h_t(x) which is the predicted value from each
        weak classifier

        Args:
            image (numpy.array): a 24x24 image which may contain a face.

        Returns:
            int: 1 or -1, 1 if there is a face -1 if not a face
        """
        
        integral = convert_image_to_integral_image(image)
        return self.predictHelper(integral)

    def faceDetection(self, image):
        """Scans for faces in a given image.
        You will want to take every 24x24 window in the input image, and check if
        it contains a face.  You will probably get several hits, so you may want to 
        combine the hits if they are nearby.  You can also consider increasing the 1/2 value
        in the inequality to reduce the number of hits (e.g. try 1/1.5)
        You should then draw a box around each face you find
        
        Args:
            image (numpy.array): Input image.
        Returns:
            an image with a box drawn around each face
        """
        h, w = image.shape[:2]
        points = np.zeros((h, w), dtype = np.float32)
        integral = convert_image_to_integral_image(image)

        for y in range(h - 24):
            for x in range(w - 24):
                if self.predictHelper(integral[y:(y + 24), x:(x + 24)]) == 1:
                    points[y, x] += 1

        points = cv2.GaussianBlur(points, (15, 15), 0)
        t = np.max(points) * 0.97
        res = np.argwhere(points > t)
        x = int(np.sum(res[:, 1]) / len(res))
        y = int(np.sum(res[:, 0]) / len(res))

        result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(result, (x, y), (x + 24, y + 24), (200, 0, 200))

        return result

class VJ_Classifier:
    """Weak classifier for Viola Jones procedure

    Args:
        X (numpy.array): Feature scores for each image. Rows: number of images
                         Columns: number of features.
        y (numpy.array): Labels array of shape (num images, )
        weights (numpy.array): observations weights array of shape (num observations, )

    Attributes:
        Xtrain (numpy.array): Feature scores, one for each image.
        ytrain (numpy.array): Labels, one per image.
        weights (float): Observations weights
        threshold (float): Integral image score minimum value.
        feat (int): index of the feature that leads to minimum classification error.
        polarity (float): Feature's sign value. Defaults to 1.
        error (float): minimized error (epsilon)
    """
    def __init__(self, X, y, weights, thresh=0, feat=0, polarity=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.weights = weights
        self.threshold = thresh
        self.feature = feat
        self.polarity = polarity
        self.error = 0

    def train(self, haar_features):
        """Trains a weak classifier that uses Haar-like feature scores.

        This process finds the feature that minimizes the error as shown in
        the Viola-Jones paper.

        Once found, the following attributes are updated:
        - feature: The column id in X.
        - threshold: Threshold (theta) used.
        - polarity: Sign used (another way to find the parity shown in the
                    paper).
        - error: lowest error (epsilon).
        """
        signs = [1] * self.Xtrain.shape[1]
        thresholds = [0] * self.Xtrain.shape[1]
        errors = [100] * self.Xtrain.shape[1]

        for f in range(self.Xtrain.shape[1]):
            tmp_thresholds = self.Xtrain[:,f].copy()
            tmp_thresholds = np.unique(tmp_thresholds)
            tmp_thresholds.sort()
            tmp_thresholds = [(tmp_thresholds[i]+tmp_thresholds[i+1])/2 for i in
                              range(len(tmp_thresholds)-1)]

            min_e = 10000000000000
            for theta in tmp_thresholds:
                for s in [1,-1]:
                    tmp_r = self.weights * ( s*((self.Xtrain[:,f]<theta)*2-1) != self.ytrain )
                    tmp_e = sum(tmp_r)
                    if tmp_e < min_e:
                        thresholds[f] = theta
                        signs[f] = s
                        errors[f] = tmp_e
                        min_e = tmp_e

        feat = errors.index(min(errors))
        self.feature = haar_features[feat]
        self.threshold = thresholds[feat]
        self.polarity = signs[feat]
        self.error = errors[feat]

    def predict(self, ii):
        """Returns a predicted label.

        Inequality shown in the Viola Jones paper for h_j(x).

        Args:
            ii (numpy.array):  integral image of the image we want to predict

        Returns:
            float: predicted label (1 or -1)
        """
        return self.polarity * ((self.feature.evaluate(ii) < self.threshold) * 2 - 1)
