class BaseOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    def step(self, closure=None):
        raise NotImplementedError




class Adam(BaseOptimizer):

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0] * len(self.params)
        self.v = [0] * len(self.params)
        self.t = 0

    def step(self, closure=None):
        self.t += 1

        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.value -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
            param.grad = 0