import os


class Mode:
    TRAIN = "train"
    TEST = "test"
    VALIDATION_WITHIN = "val_within"
    VALIDATION_UNITY = "val_unity"
    VALIDATION_MPII = "val_mpii"

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
        return value


class TranslationDirection:
    S2R = "S2R"
    R2S = "R2S"
    BOTH = "both"

    @classmethod
    def get_all(cls):
        return [cls.S2R, cls.R2S, cls.BOTH]


class DatasetClass:
    UNITY = 'unity'
    MPII = 'mpii'
    REFINED = 'refined'
