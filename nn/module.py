class Module:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input):
        raise NotImplementedError("Subclasses must implement the 'forward' method.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

