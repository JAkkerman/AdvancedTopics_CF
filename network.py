

class Network():
    def __init__(self, w, nu=1):
        self.w = w
        self.nodes = {}
        # self.nu = nu
        self.Sf = []

    def compute_R(self):
        h_T = sum([node.nu*node.h[-1] for node in self.nodes.values()])
        h_1 = sum([node.nu*node.h[0] for node in self.nodes.values()])
        R = (h_T - h_1)/(len(self.nodes.values())-1)
        return R