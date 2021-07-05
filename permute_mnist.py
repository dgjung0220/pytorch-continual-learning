import numpy as np

def permute_mnist(mnist, seed):

    np.random.seed(seed)
    print('starting permutation....')

    h = w = 28
    perm_inds = list(range(h*w))
    np.random.shuffle(perm_inds)

    perm_mnist = []

    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))

    print('done.')
    
    return perm_mnist