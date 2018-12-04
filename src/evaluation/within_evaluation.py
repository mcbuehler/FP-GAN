from evaluation.base_evaluation import Evaluation


class WithinEvaluation(Evaluation):
    """
    Trains and evaluates on refined synthetic images from UnityEyes.

    Training set: refined images from UnityEyes
    Test set: refined images from UnityEyes
    """
    def __init__(self, model_eyegaze, data_test):
        super().__init__()
        self.model_eyegaze = model_eyegaze
        self.data_test = data_test


