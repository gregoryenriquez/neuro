class Neuron:

    def __init__(self, x, y, z, n_id, c_id=-1):
        self.x = x
        self.y = y
        self.z = z
        self.n_id = n_id
        self.cluster_id = c_id

    def set_cluster(self, c):
        self.cluster_id = c

    def get_cluster(self):
        return self.cluster_id

    def info(self):
        return "id: {0}, x: {1}, y: {2}, z: {3}, cluster_id: {4}".format(self.n_id, self.x, self.y, self.z, self.cluster_id)

    def save(self):
        return "{0}, {1}, {2}, {3}, {4}\n".format(self.n_id, self.x, self.y, self.z, self.get_cluster())

    def calc_distance(self, other_neuron):
        return ( (self.x - other_neuron.x) ** 2 + 
                    (self.y - other_neuron.y) ** 2 + 
                    (self.z - other_neuron.z) ** 2 ) ** (1./2.)