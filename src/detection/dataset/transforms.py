import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class AmplitudeScale:
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, x):
        scale = np.random.uniform(self.low, self.high)
        return x * scale


class GaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        noise = np.random.normal(0.0, self.std, size=x.shape)
        return x + noise


class TimeShift:
    def __init__(self, max_shift=20):
        self.max_shift = int(max_shift)

    def __call__(self, x):
        if self.max_shift <= 0:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift, axis=1)


def build_transforms(mode):
    mode = str(mode).strip().lower()

    if mode in ["contrast", "contrastive"]:
        return Compose([
            AmplitudeScale(0.85, 1.15),
            GaussianNoise(0.01),
            TimeShift(20),
        ])

    if mode in ["reconstruction", "reconst", "recon", "cae"]:
        return Compose([
            GaussianNoise(0.01),
            TimeShift(20),
        ])

    if mode == "finetune":
        return Compose([
            AmplitudeScale(0.9, 1.1),
            GaussianNoise(0.005),
        ])

    if mode == "eval":
        return None

    return None
