import os
# from util.log import get_logger
#
# logger = get_logger()


class Mode:
    TRAIN_UNITY = "train_unity"
    TRAIN_REFINED_U2M = "train_refined_U2M"
    VALIDATION_UNITY = "val_unity"
    VALIDATION_MPII = "val_mpii"

    INFERENCE_UNITY_TO_MPII = "inf_u2mpii"
    INFERENCE_MPII_TO_UNITY = "inf_mpii2u"

    TEST_M2U = "test_mpii2u"
    SAMPLE = "sample"


class EnvironmentVariable:
    PATH_UNITY_TRAIN = "PATH_UNITY_TRAIN"
    PATH_UNITY_VALIDATION = "PATH_UNITY_VALIDATION"
    PATH_UNITY_TEST = "PATH_UNITY_TEST"

    PATH_MPII = "PATH_MPII"

    @classmethod
    def get_value(cls, variable):
        value = os.getenv(variable, None)
        # if value is None:
        #     logger.warn("Environment variable '{}' does not exist".format(variable))
        return value


class TranslationDirection:
    U2M = "U2M"
    M2U = "M2U"

    @classmethod
    def get_all(cls):
        return [cls.U2M, cls.M2U]


class DatasetClass:
    UNITY = 'unity'
    MPII = 'mpii'
    REFINED = 'refined'
