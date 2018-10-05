import sys
import numpy as np


def read_stdin():
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split('\t')[0]]

    return lines


def read_text(fn):
    lines = []
    f = open(fn, 'r')

    for line in f:
        if line.strip() != '':
            lines += [line.strip().split('\t')[0]]

    f.close()

    return lines


def get_confusion_matrix(classes, y, y_hat):
    confusion_matrix = np.zeros((len(classes), len(classes)))
    mapping_table = {}

    for idx, c in enumerate(classes):
        mapping_table[c] = idx

    for y_i, y_hat_i in zip(y, y_hat):
        confusion_matrix[mapping_table[y_hat_i], mapping_table[y_i]] += 1

    print('\t'.join(c for c in classes))
    for i in range(len(classes)):
        print('\t'.join(['%4d' % confusion_matrix[i, j] for j in range(len(classes))]))


if __name__ == '__main__':
    ref_fn = sys.argv[1]

    ref_lines = read_text(ref_fn)
    lines = read_stdin()
    
    min_length = min(len(ref_lines), len(lines))
    ref_lines = ref_lines[:min_length]
    lines = lines[:min_length]

    classes = list(set(ref_lines + lines))

    get_confusion_matrix(classes, ref_lines, lines)
