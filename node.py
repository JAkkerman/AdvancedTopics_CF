

class Node():
    def __init__(self, network, name, id, psi):
        self.network = network
        self.name = name
        self.id = id
        self.connections = {} # Saved as [(obj, w_ji)]
        self.h = [psi]
        self.s = 'U' # s={U,D,I}
        
        self.check_s()

    def check_s(self):
        if self.s == 'I':
            pass
        elif self.s == 'D':
            self.s = 'I'
        elif self.h[-1] > 0:
            self.s = 'D'
            self.network.Sf += [self]
        else:
            self.s = 'U'

    def compute_h(self, t): # Pass current t, look one back
        h_t = min(1, self.h[-1] + sum([neigh[0].h[t-1] * neigh[1] for neigh in self.connections.values()]))
        self.h += [h_t]
