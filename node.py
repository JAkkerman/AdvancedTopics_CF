

class Node():
    def __init__(self, network, name, id, psi):
        self.network = network
        self.name = name
        self.id = id
        self.connections = {} # Saved as {name:(obj, w_ij)}
        self.h = [psi]
        self.s = ['U'] # s={U,D,I}
        
        self.check_s()

    def check_s(self):
        if self.s == 'I':
            pass
        elif self.s == 'D':
            self.s += ['I']
        elif self.h[-1] > 0:
            self.s += ['D']
            self.network.Sf += [self]
        else:
            self.s += ['U']

    def compute_h(self, t): # Pass current t, look one back

        neighbor_h = []
        for neigh in self.connections.values():
            if neigh[0].s[t-1] == 'D':
               neighbor_h += [neigh[0].h[t-1] * neigh[0].connections[self.name][1]]

        # neighbor_h = sum([neigh[0].h[t-1] * neigh[0].connections[self.name][1] for neigh in self.connections.values() if neigh[0].s[t-1] == 'D'])
        h_t = min(1, self.h[t-1] + sum(neighbor_h))
        self.h += [h_t]
