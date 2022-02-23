from risk_scores import risk_score_a, risk_score_b


class ANN:

    def __init__(self):
        """ The trained optimal risk matrix W_1 and W_2"""
        self.W_1 = [
            [-0.0946, 0.0784, -1.0774, 0.4256, 0.5320, -0.8213],
            [-0.1957, 0.6868, -0.7934, 0.2466, 0.6549, 0.3644],
            [-0.1502, 0.5130, -0.0877, -0.5982, -0.6342, 0.36821]
        ]
        self.W_2 = [
            -1.1015, 0.8268, -0.7577
        ]
        self.risk_score_a = risk_score_a
        self.risk_score_b = risk_score_b

    def train(self):
        pass

    def test(self):
        pass

    def risk_assessment_ann(self):
        pass
