import os
import sys
import time
import shlex
from subprocess import Popen
from math import ceil
from cluster import Cluster
from synapse import Synapse
from random import randrange
from neuron import Neuron
from argparse import ArgumentParser

global DEBUG
global SAVE
global TIMESTAMP
DEBUG = False
SAVE = False
TIMESTAMP = int(time.time())

class Brain:
    neurons = []
    clusters = []
    edges = []
    num_clusters = 0
    num_neurons = 0
    max_value = 0 

    # def __init__(self, neurons, clusters, max_value):
    def __init__(self, *args, **kwargs):
        braindata = kwargs.get("braindata", None)
        if braindata is None:
            print("Intializing brain...")
            print("Neuron count: {0}".format(args[0]))
            print("Clusters: {0}".format(args[1]))
            self.num_neurons = args[0]
            self.num_clusters = args[1]
            self.max_value = args[2]

            for n in xrange(self.num_clusters):
                x = randrange(0, self.max_value)
                y = randrange(0, self.max_value)
                z = randrange(0, self.max_value)
                self.clusters.insert(len(self.clusters), Cluster(x, y, z, n))

            for c in self.clusters:
                print("Cluster #{0}".format(str(c.get_cluster_id())) + " Center: {0}".format(c.get_centroid_coordinates()))
        else:
            nodesfile = braindata + "/neurons.csv"
            clustersfile = braindata + "/clusters.csv"
            edgesfile = braindata + "/edges.csv"

            with open(nodesfile, "r") as nfile:
                self.num_neurons = nfile.readline().strip() # read first line count in
                for line in nfile:
                    line = line.strip()
                    ndata = line.split(", ")
                    self.neurons.append(Neuron(int(ndata[1]), int(ndata[2]), int(ndata[3]), int(ndata[0]), int(ndata[4])))

            with open(clustersfile, "r") as cfile:
                self.num_clusters = cfile.readline().strip() # read first line count in
                for line in cfile:
                    line = line.strip()
                    cdata = line.split(", ")
                    temp_cluster = Cluster(float(cdata[1]), float(cdata[2]), float(cdata[3]), int(cdata[0]))
                    for neuron in self.neurons:
                        # print(neuron.info())
                        if int(cdata[0]) == int(neuron.get_cluster()):
                            temp_cluster.add_neuron(neuron)
                    self.clusters.insert(len(self.clusters), temp_cluster)

            with open(edgesfile, "r") as efile:
                efile.readline().strip() # dump edges count
                for line in efile:
                    line = line.strip()
                    edata = line.split(", ")
                    self.edges.append([int(edata[0]), int(edata[1])])

    def generate_neurons(self, distribution_type, max_val):
        print("Max neuron dimension value: {0}".format(max_val))
        if distribution_type == "clustered":
            # seed first few values, 10% of total
            # use seeds as centroids for clusters
            centers = self.num_neurons / 10
            centers_list = []
            for i in xrange(centers):
                x = randrange(0, max_val)
                y = randrange(0, max_val)
                z = randrange(0, max_val)
                temp_neur = Neuron(x, y, z, -1)
                centers_list.append(temp_neur)

            for i in xrange(self.num_neurons):
                max_inter_cluster_distance = max_val / 10
                center_choice = randrange(0, len(centers_list))
                x = centers_list[center_choice].x + randrange(0, max_inter_cluster_distance)
                y = centers_list[center_choice].y + randrange(0, max_inter_cluster_distance)
                z = centers_list[center_choice].z + randrange(0, max_inter_cluster_distance)
                self.neurons.append(Neuron(x, y, z, i))

        elif distribution_type == "uniform":
            counter = 0
            num_per_dimension = ceil(self.num_neurons ** (1./3.))
            self.num_neurons = int(num_per_dimension) ** 3
            print("Adjusting for uniform, neuron count now: " + str(self.num_neurons))
            distance = float(max_val)/int(num_per_dimension - 1) #include [0, 0, 0]
            if DEBUG:
                print("Number per dimension: " + str(num_per_dimension))
                print("Distance between neurons: " + str(distance))
            for x in xrange(int(num_per_dimension)):
                for y in xrange(int(num_per_dimension)):
                    for z in xrange(int(num_per_dimension)):
                        self.neurons.append(Neuron(x*distance, y*distance, z*distance, counter))
                        counter += 1

        elif distribution_type == "random":
            for i in xrange(self.num_neurons):
                x = randrange(0, max_val)
                y = randrange(0, max_val)
                z = randrange(0, max_val)
                self.neurons.append(Neuron(x, y, z, i))

        # if DEBUG:
        #     for i in xrange(len(self.neurons)):
        #         print(self.neurons[i].info())

    def generate_synapses(self, intra_prob, inter_prob, inter_thresh, intra_thresh):
        # N^N time
        # reduced to N! time

        # define inner_threshold
        intra_thresh = float(intra_thresh)/float(100) * self.max_value
        inter_thresh = float(inter_thresh)/float(100) * self.max_value

        for i in xrange(len(self.neurons)):
            for j in xrange(len(self.neurons[i:len(self.neurons)])):
                dist = self.neurons[i].calc_distance(self.neurons[j])
                dice = randrange(0, 100)
                dice_back = randrange(0, 100)
                if dist < intra_thresh:
                    if dice > intra_prob:
                        self.edges.append([i, j + i])
                        # self.edges.append(Synapse(i, j + i))
                    if dice_back > intra_prob:
                        self.edges.append([j + i, i])
                        # self.edges.append(Synapse(j + i, i))
                elif dist < inter_thresh:
                    if dice > inter_prob:
                        self.edges.append([i, j+ i])
                        # self.edges.append(Synapse(i, j + i))
                    if dice_back > inter_prob:
                        self.edges.append([j + i, i])
                        # self.edges.append(Synapse(j + i, i))

        # if DEBUG:
        #     for i in xrange(len(self.edges)):
        #         print(self.edges[i])

        # self.edges.insert(0, [0, 1])
        # self.edges.insert(0, [1, 0])
        # self.edges.insert(0, [0, 2])
        # self.edges.insert(0, [2, 0])
        # self.edges.insert(0, [1, 2])
        # self.edges.insert(0, [2, 1])

    def k_cluster_neurons(self, iterations):
        for iter_ in xrange(iterations):
            for n in self.neurons:
                distances = []
                old_cluster_id = n.get_cluster()
                for c in self.clusters:
                    distance = n.calc_distance(c)
                    # print("Distance to cluster#: " + str(c.get_cluster_id()) + " distance: " + str(distance))
                    distances.insert(len(distances), distance)
                index = distances.index(min(distances)) # cluster of index with min value

                n.set_cluster(index)
                # print("Assigning: " + n.info() + " to cluster#: " + str(index))
                if old_cluster_id == -1:
                    # n.set_cluster(index)
                    self.clusters[index].add_neuron(n)                    
                elif old_cluster_id == index:
                    pass
                else:
                    # n.set_cluster(index)
                    self.clusters[old_cluster_id].move_neuron(n, self.clusters[index])
                    # self.clusters[index].move_neuron(n, self.clusters[old_cluster_id])

            for c in self.clusters:
                c.center_centroid()

            for c in self.clusters:
            # c.print_neurons()
                print(c.info())

            print("----------------------------------------")

    def save_brain(self):
        with open(str(TIMESTAMP) + "/neurons.csv", "w") as neuronfile:
            neuronfile.write(str(self.num_neurons) + "\n")
            for n in self.neurons:
                neuronfile.write(n.save())

        with open(str(TIMESTAMP) + "/clusters.csv", "w") as clusterfile:
            clusterfile.write(str(self.num_clusters) + "\n")
            for c in self.clusters:
                clusterfile.write(c.save())

        with open(str(TIMESTAMP) + "/edges.csv", "w") as edgefile:
            edgefile.write(str(len(self.edges)) + "\n")
            for e in self.edges:
                edgefile.write("{0}, {1}\n".format(e[0], e[1]))

        print("Saved to {0}\ directory".format(str(TIMESTAMP)))

    def count_all_triangles(self):

        brain_size = len(self.neurons)
        cube_size = (brain_size * (brain_size - 1) * (brain_size - 2)) / 6
        counter = 0
        i = 0
        percentage = 0
        diff = 0

        try:
            for n1 in self.neurons:
                for n2 in self.neurons[n1.n_id:brain_size]:
                    for n3 in self.neurons[n2.n_id:brain_size]:
                        if self.is_a_triangle(n1, n2, n3):
                            counter += 1

                        if DEBUG is True:
                            # sys.stdout.write("\rTriangles: " + str(counter))
                            # sys.stdout.flush()
                            percentage = float(i)/float(cube_size) * 100
                            diff = 100 - percentage
                            sys.stdout.write("\r[" + "#" * int(percentage) + " " * int(diff) + "]" + "{0:.2f}".format(percentage) + "%")                
                            sys.stdout.flush()
                            i += 1
            print("\n")
        except KeyboardInterrupt:
            print("\n")
            return counter
        return counter

    def is_a_triangle(self, n1, n2, n3):
        if [n1.n_id, n2.n_id] in self.edges and [n2.n_id, n1.n_id] in self.edges:
            if [n1.n_id, n3.n_id] in self.edges and [n1.n_id, n3.n_id] in self.edges:
                if [n2.n_id, n3.n_id] in self.edges and [n3.n_id, n2.n_id] in self.edges:
                    return True
        return False

    def info(self):
        return "Neurons: " + str(len(self.neurons)) + " Synapses: " + str(len(self.edges)) + " Clusters: " + str(len(self.clusters))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-k", dest="clusters", type=int)
    parser.add_argument("-n", dest="neurons", type=int)
    parser.add_argument("-m", dest="maxvalue", type=int)
    parser.add_argument("-i", dest="iterations", type=int)
    parser.add_argument("--load-brain", dest="brain_data_dir", type=str, default=None, help="timestamp dir/")
    parser.add_argument("--save-brain-nodes-and-edges", dest="persist", action="store_true", default=False)
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    parser.add_argument("--neuron-distribution", dest="neuron_distribution_type", choices=["clustered", "uniform", "random"])
    parser.add_argument("--inter-cluster-edge-prob-perct", dest="intercluster_edge_probability", type=int)
    parser.add_argument("--intra-cluster-edge-prob-perct", dest="intracluster_edge_probability", type=int)
    parser.add_argument("--inter-cluster-threshold-percentage", dest="intercluster_thresh", type=int)
    parser.add_argument("--intra-cluster-threshold-percentage", dest="intracluster_thresh", type=int)
    options = parser.parse_args()
    DEBUG = options.debug
    SAVE = options.persist

    if SAVE is True:
        cmd = "mkdir {0}".format(TIMESTAMP)
        Popen(shlex.split(cmd))

    if options.brain_data_dir is not None:
        brain = Brain(braindata=options.brain_data_dir)

    if options.brain_data_dir is None:
        neurons = options.neurons
        clusters = options.clusters
        maxvalue = options.maxvalue
        brain = Brain(neurons, clusters, maxvalue)
        brain.generate_neurons(options.neuron_distribution_type, options.maxvalue)
        brain.generate_synapses(options.intracluster_edge_probability, options.intercluster_edge_probability,
                                options.intercluster_thresh, options.intracluster_thresh)
        brain.k_cluster_neurons(options.iterations)

    if SAVE is True:
        brain.save_brain()

    print(brain.info())
    start_time = float(time.time())
    print("Counting ALL (N^3 time) triangles... this may take a while")
    triangles = brain.count_all_triangles()
    print("Triangles: " + str(triangles))
    print("Runtime: {0} seconds".format(str(float(time.time()) - float(start_time) / 1000)))
