

class Node():
    def __init__(self, name, id, connections):
        self.name = name
        self.id = id
        self.connections = connections
        self.has_defaulted = False

    def check_default(self):
        return False