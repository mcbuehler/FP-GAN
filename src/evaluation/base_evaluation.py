


class Evaluation:
    def __init__(self):
        pass

    def create_training_set(self):
        raise NotImplementedError("Implement this in a subclass")

    def create_test_set(self):
        raise NotImplementedError("Implement this in a subclass")

    def run(self):
        raise NotImplementedError("Implement this in a subclass")