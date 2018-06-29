import numpy as np
import traceback


def get_mean_std_channels(dataloader):
    try:
        pop_mean = []
        pop_std = []
        for i, data in enumerate(dataloader, 0):
            np_img = data[0].numpy()
            batch_mean = np.mean(np_img, axis=(0, 2, 3))
            batch_std = np.std(np_img, axis=(0, 2, 3))

            pop_mean.append(batch_mean)
            pop_std.append(batch_std0)

        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_std = np.array(pop_std0).mean(axis=0)
        return pop_mean, std
    except Exception as e:
        print(traceback.format_exc())
        raise e
