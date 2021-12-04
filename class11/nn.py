class NeuralNetwork:
    """
    Back-Propagation Neural Networks
    Placed in the public domain.
    Original author: Neil Schemenauer <nas@arctrix.com>
    Modified by: Massimo Di Pierro
    Read more: http://www.ibm.com/developerworks/library/l-neural/
    """

    @staticmethod
    def rand(a, b):
        """calculate a random number where:  a <= rand < b"""
        return (b - a) * random.random() + a

    @staticmethod
    def sigmoid(x):
        """our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)"""
        return math.tanh(x)

    @staticmethod
    def dsigmoid(y):
        """# derivative of our sigmoid function, in terms of the output"""
        return 1.0 - y ** 2

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = Matrix(self.ni, self.nh, fill=lambda r, c: self.rand(-0.2, 0.2))
        self.wo = Matrix(self.nh, self.no, fill=lambda r, c: self.rand(-2.0, 2.0))

        # last change in weights for momentum
        self.ci = Matrix(self.ni, self.nh)
        self.co = Matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError("wrong number of inputs")

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            s = sum(self.ai[i] * self.wi[i, j] for i in range(self.ni))
            self.ah[j] = self.sigmoid(s)

        # output activations
        for k in range(self.no):
            s = sum(self.ah[j] * self.wo[j, k] for j in range(self.nh))
            self.ao[k] = self.sigmoid(s)
        return self.ao[:]

    def back_propagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError("wrong number of target values")

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = self.dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = sum(output_deltas[k] * self.wo[j, k] for k in range(self.no))
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j, k] = self.wo[j, k] + N * change + M * self.co[j, k]
                self.co[j, k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i, j] = self.wi[i, j] + N * change + M * self.ci[i, j]
                self.ci[i, j] = change

        # calculate error
        error = sum(0.5 * (targets[k] - self.ao[k]) ** 2 for k in range(len(targets)))
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], "->", self.update(p[0]))

    def weights(self):
        print("Input weights:")
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print("Output weights:")
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, check=False):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, N, M)
            if check and i % 100 == 0:
                print("error %-14f" % error)

