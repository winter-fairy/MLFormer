import numpy as np
from sklearn.metrics import average_precision_score


class VOCmAP(object):
    def __init__(self):
        super().__init__()
        self.y_true = None
        self.y_pred = None

    def update(self, y_true, y_pred):
        """
        store all the true and predicted labels for a dataset
        :param y_true: [batch_size, num_classes], numpy array, dtype=np.float32
        :param y_pred: [batch_size, num_classes], numpy array, dtype=np.float32
        :return:
        """
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.vstack((self.y_true, y_true))

        if self.y_pred is None:
            self.y_pred = y_pred
        else:
            self.y_pred = np.vstack((self.y_pred, y_pred))

    def get_aps(self):
        """
        calculate ap for each class based on the stored true and predicted labels
        :return:
        """
        aps = []
        for i in range(20):
            class_pred = self.y_pred[:, i]
            class_true = self.y_true[:, i]

            ap = average_precision_score(class_true, class_pred)
            aps.append(ap)
        return aps

    def reset(self):
        """
        clean all the stored data
        """
        self.y_true = None
        self.y_pred = None
