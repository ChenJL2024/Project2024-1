import numpy as np
import math

## 平滑关机点，使关键点稳定下来，不跳动

class KeypointSmoothing(object):
    # The following code are modified from:
    # https://github.com/jaantollander/OneEuroFilter

    def __init__(self,
                 width,
                 height,
                 filter_type,
                 alpha=0.5,
                 fc_d=0.1,
                 fc_min=0.1,
                 beta=0.1,
                 thres_mult=0.3):
        super(KeypointSmoothing, self).__init__()
        self.image_width = width
        self.image_height = height
        self.threshold = np.array([
            0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
        ]) * thres_mult
        self.filter_type = filter_type
        self.alpha = alpha
        self.dx_prev_hat = None
        self.x_prev_hat = None
        self.fc_d = fc_d
        self.fc_min = fc_min
        self.beta = beta

        if self.filter_type == 'OneEuro':
            self.smooth_func = self.one_euro_filter
        elif self.filter_type == 'EMA':
            self.smooth_func = self.ema_filter
        else:
            raise ValueError('filter type must be one_euro or ema')

    def smooth_process(self, current_keypoints):
        if self.x_prev_hat is None:
            self.x_prev_hat = current_keypoints[:, :2]
            self.dx_prev_hat = np.zeros(current_keypoints[:, :2].shape)
            return current_keypoints
        else:
            result = current_keypoints
            num_keypoints = len(current_keypoints)
            for i in range(num_keypoints):
                result[i, :2] = self.smooth(current_keypoints[i, :2],
                                            self.threshold[i], i)
            return result

    def smooth(self, current_keypoint, threshold, index):
        distance = np.sqrt(
            np.square((current_keypoint[0] - self.x_prev_hat[index][0]) /
                      self.image_width) + np.square((current_keypoint[
                          1] - self.x_prev_hat[index][1]) / self.image_height))
        if distance < threshold:
            result = self.x_prev_hat[index]
        else:
            result = self.smooth_func(current_keypoint, self.x_prev_hat[index],
                                      index)

        return result

    def one_euro_filter(self, x_cur, x_pre, index):
        te = 1
        self.alpha = self.smoothing_factor(te, self.fc_d)
        dx_cur = (x_cur - x_pre) / te
        dx_cur_hat = self.exponential_smoothing(dx_cur, self.dx_prev_hat[index])

        fc = self.fc_min + self.beta * np.abs(dx_cur_hat)
        self.alpha = self.smoothing_factor(te, fc)
        x_cur_hat = self.exponential_smoothing(x_cur, x_pre)
        self.dx_prev_hat[index] = dx_cur_hat
        self.x_prev_hat[index] = x_cur_hat
        return x_cur_hat

    def ema_filter(self, x_cur, x_pre, index):
        x_cur_hat = self.exponential_smoothing(x_cur, x_pre)
        self.x_prev_hat[index] = x_cur_hat
        return x_cur_hat

    def smoothing_factor(self, te, fc):
        r = 2 * math.pi * fc * te
        return r / (r + 1)

    def exponential_smoothing(self, x_cur, x_pre, index=0):
        return self.alpha * x_cur + (1 - self.alpha) * x_pre