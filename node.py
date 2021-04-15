

class Node():
    def __init__(self, eta_0):
        self.eta_0 = eta_0
        self.eta_t = eta_0
        self.chi = 0
        self.borrowers = []
        self.lenders = []
        self.k_fi = 0
        self.assets = {}
        self.liabilities = {}
        self.has_defaulted = False

    def check_default(self):
        return False

    def update_kfi(self):
        kfi = sum([1 for borrower in self.borrowers if borrower.has_defaulted])
        self.kfi = kfi

    def update_eta(self):
        pass