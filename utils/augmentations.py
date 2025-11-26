import numpy as np

def strong_augment(x, max_shift=3, drop_prob=0.3, scale_range=(0.6, 1.4), noise_std=0.05):
    """
    Apply strong data augmentations to an input matrix.
    Includes random shift, dropout, scaling, and Gaussian noise.

    Returns:
        np.ndarray: Augmented matrix (same shape as input)
    """
    # 1️ Random shift along sequence axis
    s = np.random.randint(-max_shift, max_shift + 1)
    x = np.roll(x, shift=s, axis=0)

    # 2️ Random dropout mask
    mask = np.random.rand(*x.shape) > drop_prob
    for i in range(x.shape[0]):
        if not mask[i].any():
            mask[i, np.random.randint(0, x.shape[1])] = 1
    x *= mask

    # 3️ Random scaling per feature channel
    scale = np.random.uniform(scale_range[0], scale_range[1], size=(1, x.shape[1]))
    x *= scale

    # 4️ Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)
    x += noise

    return x.astype(np.float32)
