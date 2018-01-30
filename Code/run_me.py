"""
    Facial Recognition using PCA
    Math 545    
"""
from pca import PCA
from image_control import IM_CONTROL # image control
import numpy as np
import sys
import os
from collections import defaultdict
import random
import time
from plotting import *



def main(matrix, pcomps):
    pca = PCA(pcomps, matrix)
    # get covariance matrix - saved as instance var
    pca.covariance_matrix()
    # find evecs and evals of covariance matrix
    pca.evecs_and_evals()
    # get top x principal components aka evecs corresponding to top evals
    pca.principal_components()
    # reduce image on those components
    return pca.get_evec_matrix()



class Train:
    def __init__(self, pcomps):
        self.test = list()
        self.train = list()
        self.training = list()
        self.mu = None
        self.pcomps = pcomps

    def mean_of_images(self):
        """
        Returns mean grey-scale matrix of all images.
        """
        ctr = 0
        test = dict()
        store = np.zeros((112, 92))
        for image in self.train:
            image_manipulation = IM_CONTROL(image)
            matrix = image_manipulation.grey_scale_matrix() # get greyscale matrix
            store = store + matrix
            ctr += 1
        self.mu = (1/float(ctr))*store # average of all faces
        image_manipulation.show_image(self.mu)

    def read_data(self, train_size):
        """
        Splits the 400 images into training and test sets depending on size given.
        """
        subjects = [i for i in os.listdir("att_faces") if i.startswith("s")]
        images = list()
        for subject in subjects:
            per_sub = os.listdir("att_faces/" + subject)
            list_of_images = ["att_faces/" + subject + "/" + pic for pic in per_sub if pic.endswith("m")]
            sorted(list_of_images)
            random.seed(3)
            random.shuffle(list_of_images)
            images.append(list_of_images)
        for row in range(len(images)):
            self.training.append(images[row][:train_size])
            for i in range(train_size):
                self.train.append(images[row][i])
            for i in range(1,11-train_size):
                self.test.append(images[row][-i])

    def train_data(self):
        """
        Assigns weight to each person given their evec matrices.
        """
        sub_training = defaultdict(list)
        for row in self.training:
            subject = row[0].split("/")[1]
            for image in row:
                image_manipulation = IM_CONTROL(image)
                grey_scale = image_manipulation.grey_scale_matrix()
                x_minus_mu = grey_scale - self.mu
                evec_test = main(x_minus_mu, self.pcomps)
                weights = np.dot(evec_test.transpose(), x_minus_mu)
                sub_training[subject].append((evec_test, weights))
        return sub_training, self.test, self.mu



class Test:
    def __init__(self, training_data, test_data, mu, pcomps):
        self.training_data = training_data
        self.mu = mu
        self.pcomps = [pcomps]
        self.test_identity = None
        self.test_data = test_data

    def classify(self):
        num, den, top = 0, 0, 0
        for pcomps in self.pcomps:
            for image in self.test_data:
                self.test_identity = image.split("/")[1]
                image_manipulation = IM_CONTROL(image)
                grey_scale = image_manipulation.grey_scale_matrix()
                x_minus_mu = grey_scale - self.mu  # subtract mean from given image matrix
                evec_test = main(x_minus_mu, pcomps)  # compute evec given specified # of pcomps
                
                distances = dict()
                for k, v in self.training_data.items():
                    weights = np.dot(v[0][0].transpose(), x_minus_mu)
                    weights = weights + weights.min()
                    weights2 = v[0][1] + v[0][1].min()
                    # compute difference between each training set individuals weights
                    # projected on given image matrix and this guy's projection
                    x = weights - weights2 
                    distances[np.linalg.norm(x)] = k # store distance between matrices
                if distances[min(distances.keys())] == self.test_identity:
                    num += 1 # if top match == person => success
                sorted_distances = sorted(distances.keys())
                top_matches = [distances[i] for i in sorted_distances[:5]]
                if self.test_identity in top_matches:
                    top += 1 # if top match is in top 5 matches
                den += 1
        return pcomps, num/float(den), top/float(den)


if __name__=='__main__':
    highest_accuracies = dict()
    # Test recognition over different train-test splits and different #s of principal components
    for train_size in [2,4,6,8]:
        print "Train/Test Ratio: {0}:{1}".format(train_size, 10-train_size)
        highest_accuracy = 0
        accuracy_per_comps = dict()
        time_per_comps = dict()
        ball_park_per_comps = dict()
        for pcomps in [12,22,32,42,52,62,72,82,92,112]:
            train = Train(pcomps)
            train.read_data(train_size)
            train.mean_of_images()
            
            training_data, test_data, mu = train.train_data()
            test = Test(training_data, test_data, mu, pcomps)
            start = time.time()
            pcomps, accuracy, ball_park = test.classify()
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                highest_accuracies[train_size*40] = accuracy*100.0

            accuracy_per_comps[pcomps] = accuracy*100.0
            time_per_comps[pcomps] = time.time() - start
            ball_park_per_comps[pcomps] = ball_park*100.0
        # EVERYTHING BELOW PLOTS GRAPHS
        if train_size == 8:
            x = sorted(accuracy_per_comps)
            y = [accuracy_per_comps[i] for i in x]
            single_line("Recognition Accuracy vs # of Principal Components (80-20, Train-Test Split)", x, y, "# of Principal Components", "Recognition Accuracy (%)", 0, 100, 12, 112, "accuracy_v_components_8_2_split.png")

            x = sorted(time_per_comps)
            y = [time_per_comps[i] for i in x]
            single_line("Time (seconds) to Classify Face vs # of Principal Components (80-20, Train-Test Split)", x, y, "# of Principal Components", "Time (seconds) for Classification", 2, 4, 12, 112, "time_v_pcomps_8_2_split.png")
            
            x = sorted(ball_park_per_comps)
            y = [ball_park_per_comps[i] for i in x]
            single_line("Narrowed Down to 12% of People vs # of Principal Components (80-20 Split)", x, y, "# of Principal Components", "Successfully Narrowed Down Matches (%)", 0, 100, 12, 112, "narrowed_down_v_pcomps_8_2_split.png")

    single_line("Highest Recognition Accuracy vs Training Data Size", highest_accuracies.keys(), highest_accuracies.values(), "Size of Training Data", "Recognition Accuracy (%)", 0, 100, 80, 320, "accuracy_v_training_size.png")
    







