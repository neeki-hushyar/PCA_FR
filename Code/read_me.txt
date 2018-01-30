read_me.txt

To run program:
	Python2.7 run_me.py
* Note must b


This program takes a set of images to train on, and then a test of images whose identities must be identified. The results are plotted in the following .png files:

accuracy_v_components_8_2_split.png: plots recognition rate over a range of numbers of principal components

time_v_pcomps_8_2_split.png: plots the time needed to classify an image, based on the number of principal components -> size of weights matrix making up the training set, over a range of numbers of principal components

narrowd_down_v_pcomps_8_2_splot.png: plots success rate at narrowing down the pool of potential candidates to 5 candidates over range of numbers of principal components


The following modules were written to achieve this:

pca.py:
	returns set of principal components, aka eigenvector matrix where each eigenvector, corresponds to the largest eigenvalue, in descending order. The size of the eigenvector matrix corresponds to the number of principal components this class is initialized with.

image_control.py:
	Can return a grayscale matrix of an image or take a grayscale image and save it as a .png.

run_me.py:
	Handles training and testing phases. The execution is tested over a range of principal component values and a range of train-test split break-downs.
