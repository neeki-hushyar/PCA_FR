# highest accuracy rate per size of training data

# 8:2
# accuracy per # of components
# narrowed down accuracy per # of components

# time per # of components
import numpy as np
import matplotlib.pyplot as plt

def single_line(title, x, y, x_label, y_label, y_low, y_high, x_low, x_high, save_to):
    plt.clf()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ind = np.arange(len(x))
    plt.plot(x, y, 'sb-', linewidth=3)
    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)
    plt.savefig(save_to)
