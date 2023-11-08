class BaseMetric:
    def __init__(self, name=None, ignore_on_eval=False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.ignore_on_eval = ignore_on_eval

    def __call__(self, **batch):
        raise NotImplementedError()
