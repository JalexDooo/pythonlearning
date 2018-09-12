import numpy as np


def onehot(labels):
	n_sample = len(labels)
	n_calss = max(labels)+1
	onehot_labels = np.zeros((n_sample, n_calss))
	onehot_labels[np.arange(n_sample), labels] = 1
	return onehot_labels


if __name__ == '__main__':
	labels = [1, 3, 2, 0, 6, 4]
	print(onehot(labels))