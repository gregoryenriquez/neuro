class Cluster:

    def __init__(self, x, y, z, c_id):
        self.x = x
        self.y = y
        self.z = z
        self.c_id = c_id
        self.neurons = []
        self.cluster_size = 0

    def add_neuron(self, neuron):
        # print("added neuron: " + neuron.info())
        self.neurons.append(neuron)
        self.cluster_size += 1
        # print("counter: " + str(self.counter))

    def remove_neuron(self, neuron):
        # print("removed neuron: " + neuron.info())
        self.neurons.remove(neuron)
        self.cluster_size -= 1
        # self.print_neurons()

    def save(self):
        list_as_str = ""
        neuron_size = len(self.neurons)
        for i in xrange(neuron_size):
            if i != neuron_size - 1:
                list_as_str = list_as_str + str(self.neurons[i].n_id) + "-"
            else:
                list_as_str = list_as_str + str(self.neurons[i].n_id)
        return "{0}, {1}, {2}, {3}, {4}, {5}\n".format(self.c_id, self.x, self.y, self.z, self.cluster_size, list_as_str)

    def load_neurons(neuron_id_list):
        # TODO
        pass

    def get_centroid_coordinates(self):
        return list([self.x, self.y, self.z])

    def get_cluster_id(self):
        return self.c_id

    def get_cluster_size(self):
        return self.cluster_size

    def get_neurons(self):
        return self.neurons

    def print_neurons(self):
        print("Cluster# " + str(self.c_id))
        for n in self.neurons:
            print("id: " + str(n.n_id) + " X: " + str(n.x) + "\tY: " + str(n.y) + "\tZ: " + str(n.z) + "\tCID: " + str(n.cluster_id))

    def info(self):
        return "CID: " + str(self.c_id) + " Size: " + str(len(self.neurons)) + " Center: [{0},{1},{2}]".format(self.x, self.y, self.z)

    def center_centroid(self):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for n in self.neurons:
            sum_x += n.x
            sum_y += n.y
            sum_z += n.z

        if sum_x + sum_y + sum_z == 0.0:
            return
        self.x = float(sum_x) / float(len(self.neurons))
        self.y = float(sum_y) / float(len(self.neurons))
        self.z = float(sum_z) / float(len(self.neurons))

    def move_neuron(self, neuron, other_cluster):
        temp_neuron = neuron
        other_cluster.add_neuron(temp_neuron)
        self.remove_neuron(neuron)

