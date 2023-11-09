class BaseMetric:
    def __init__(self, name=None, skip_on_test=False, skip_on_train=False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.skip_on_test = skip_on_test
        self.skip_on_train = skip_on_train

    def __call__(self, **batch):
        raise NotImplementedError()
