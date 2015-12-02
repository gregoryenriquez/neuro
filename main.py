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
global EDGES_DICT
DEBUG = False
SAVE = False
TIMESTAMP = int(time.time())
EDGES_DICT = {}

class Brain:
    neurons = []
    clusters = []
    edges = []
    num_clusters = 0
    num_neurons = 0
    max_value = 0 

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

    def generate_neurons(self, distribution_type, max_val, clusters):
        print("Max neuron dimension value: {0}".format(max_val))
        if distribution_type == "clustered":
            # seed first few values, 10% of total
            # use seeds as centroids for clusters
            centers = self.num_neurons / clusters
            centers_list = []
            for i in xrange(centers):
                x = randrange(0, max_val)
                y = randrange(0, max_val)
                z = randrange(0, max_val)
                temp_neur = Neuron(x, y, z, -1)
                centers_list.append(temp_neur)

            temp_dict = {}
            max_inter_cluster_distance = int(max_val * 0.03)
            print("Max intra cluster distance: " + str(max_inter_cluster_distance))
            for i in xrange(self.num_neurons):
                center_choice = randrange(0, len(centers_list))
                x_lower_bound = centers_list[center_choice].x - max_inter_cluster_distance
                if x_lower_bound < 0:
                    x_lower_bound = 0

                y_lower_bound = centers_list[center_choice].y - max_inter_cluster_distance
                if y_lower_bound < 0:
                    y_lower_bound = 0

                z_lower_bound = centers_list[center_choice].z - max_inter_cluster_distance
                if z_lower_bound < 0:
                    z_lower_bound = 0

                x = randrange(x_lower_bound, centers_list[center_choice].x + max_inter_cluster_distance)
                y = randrange(y_lower_bound, centers_list[center_choice].y + max_inter_cluster_distance)
                z = randrange(z_lower_bound, centers_list[center_choice].z + max_inter_cluster_distance)

                attempts = 0
                while (x, y, z) in temp_dict.keys():
                    x = randrange(x_lower_bound, centers_list[center_choice].x + max_inter_cluster_distance)
                    y = randrange(y_lower_bound, centers_list[center_choice].y + max_inter_cluster_distance)
                    z = randrange(z_lower_bound, centers_list[center_choice].z + max_inter_cluster_distance)
                    attempts += 1
                    if attempts > 10:
                        print("WARN: Multiple attempts getting random dupe neuron")
                    if attempts > 100:
                        print("Quitting due to possible infinite loop when generating neurons")
                        sys.exit(1)
                
                temp_dict[(x, y, z)] = True
                self.neurons.append(Neuron(x, y, z, i))

        elif distribution_type == "uniform":
            counter = 0
            num_per_dimension = ceil(self.num_neurons ** (1./3.))
            self.num_neurons = int(num_per_dimension) ** 3
            print("Adjusting for uniform, neuron count now: " + str(self.num_neurons))
            distance = float(max_val)/int(num_per_dimension) 
            if DEBUG:
                print("Number per dimension: " + str(num_per_dimension))
                print("Distance between neurons: " + str(distance))
            for x in xrange(int(num_per_dimension)):
                for y in xrange(int(num_per_dimension)):
                    for z in xrange(int(num_per_dimension)):
                        self.neurons.append(Neuron((x+1)*distance, (y+1)*distance, (z+1)*distance, counter))
                        counter += 1

        elif distribution_type == "random":
            for i in xrange(self.num_neurons):
                x = randrange(0, max_val)
                y = randrange(0, max_val)
                z = randrange(0, max_val)
                self.neurons.append(Neuron(x, y, z, i))


    def generate_synapses(self, intra_prob, inter_prob, intra_thresh, inter_thresh):
        # N^N time
        # reduced to N! time
        # 10,000 to 1 ratio

        # define euclidean distance thresholds
        intra_thresh = (float(intra_thresh)/float(100) * self.max_value) ** (3.0/2.0)
        inter_thresh = (float(inter_thresh)/float(100) * self.max_value) ** (3.0/2.0)
        print("Euclidean threshold distance for intracluster(Pr={0}%): {1}".format(str(intra_prob), str(intra_thresh)))
        print("Euclidean threshold distance for intercluster(Pr={0}%): {1}".format(str(inter_prob), str(inter_thresh)))
        total_intra = 0 
        total_inter = 0
        for i in xrange(len(self.neurons)):
            for j in xrange(len(self.neurons[i:len(self.neurons)])):
                dist = self.neurons[i].calc_distance(self.neurons[j+i])
                dice = randrange(0, 100)
                dice_back = randrange(0, 100)
                if dist < intra_thresh:
                    if dice <= intra_prob:
                        total_intra += 1
                        self.edges.append([i, j + i])
                    if dice_back <= intra_prob:
                        total_intra += 1
                        self.edges.append([j + i, i])
                elif dist < inter_thresh:
                    if dice <= inter_prob:
                        total_inter += 1
                        self.edges.append([i, j + i])
                    if dice_back <= inter_prob:
                        total_inter += 1
                        self.edges.append([j + i, i])

        print("Total intra edges: {0}".format(str(total_intra)))
        print("Total inter edges: {0}".format(str(total_inter)))

    def k_cluster_neurons(self, iterations):
        for iter_ in xrange(iterations):
            for n in self.neurons:
                distances = []
                old_cluster_id = n.get_cluster()
                for c in self.clusters:
                    distance = n.calc_distance(c)
                    distances.insert(c.c_id, distance)
                index = distances.index(min(distances)) # cluster of index with min value

                n.set_cluster(index)
                if old_cluster_id == -1:
                    self.clusters[index].add_neuron(n)                    
                elif old_cluster_id == index:
                    pass
                else:
                    self.clusters[old_cluster_id].move_neuron(n, self.clusters[index])

            for c in self.clusters:
                c.center_centroid()

    def generate_edges_map(self):
        edges_dict = {}
        for e in self.edges:
            tup = (e[0], e[1])
            edges_dict[tup] = True

        return edges_dict

    def print_cluster_info(self):
        for c in self.clusters:
            print(c.info())

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

    def save_result(self, result):
        with open(str(TIMESTAMP) + "/results.txt", "a") as results_file:
            results_file.write(str(result) + "\n")

    def count_cluster_triangles(self):
        counter = 0
        total_loops_count = 0
        for c in self.clusters:
            cluster_size = c.get_cluster_size()
            if cluster_size == 0:
                continue
            print("Cluster #{0} Size: {1}".format(str(c.get_cluster_id()), str(cluster_size)))
            cubed_size = cluster_size * (cluster_size - 1) * (cluster_size - 2)
            cubed_size = cubed_size / 6

            i = 0
            percentage = 0
            diff = 0
            c_id = c.get_cluster_id()
            neurons = c.get_neurons()

            try:
                for n in xrange(len(neurons)): #0 to max
                    for j in xrange(len(neurons[n+1:cluster_size])): #i to max
                        for z in xrange(len(neurons[n+j+2:cluster_size])): #j+i to max
                            n1 = neurons[n]
                            n2 = neurons[j+n+1]
                            n3 = neurons[z+n+j+2]
                            if self.is_a_triangle(n1, n2, n3):
                                counter += 1
                            if DEBUG is True:
                                percentage = float(i)/float(cubed_size) * 100
                                diff = 100 - percentage
                                sys.stdout.write("\r[" + "#" * int(percentage) + " " * int(diff) + "]" + "{0:.2f}".format(percentage) + "%")                
                                sys.stdout.flush()
                            i += 1
                            total_loops_count +=1
                print("\n")

            except KeyboardInterrupt:
                print("\n")
                return counter
            except:
                print("\n")
                print(counter)
                raise
        if DEBUG is True:
            print("Total cluster-count operations: " + str(total_loops_count))
        return counter

    def count_all_triangles(self):
        brain_size = len(self.neurons)
        cube_size = brain_size * (brain_size - 1) * (brain_size - 2)
        cube_size = cube_size / 6
        counter = 0
        i = 0
        percentage = 0
        diff = 0

        try:
            for n1 in self.neurons:
                for n2 in self.neurons[n1.n_id + 1:brain_size]:
                    for n3 in self.neurons[n2.n_id + 1:brain_size]:
                        if self.is_a_triangle(n1, n2, n3):
                            counter += 1

                        if DEBUG is True:
                            percentage = float(i)/float(cube_size) * 100
                            diff = 100 - percentage
                            sys.stdout.write("\r[" + "#" * int(percentage) + " " * int(diff) + "]" + "{0:.2f}".format(percentage) + "%")                
                            sys.stdout.flush()
                        i += 1
            print("\n")
        except KeyboardInterrupt:
            print("\n")
            return counter
        except: 
            print("\n")
            print(counter)
            raise
        print("Total all-count operations: " + str(i))
        return counter

    def is_a_triangle(self, n1, n2, n3):
        # each edge in dict is 24 bytes, .024kb
        try:
            return EDGES_DICT[(n1.n_id, n2.n_id)] and EDGES_DICT[(n2.n_id, n1.n_id)] and EDGES_DICT[(n1.n_id, n3.n_id)] and \
                    EDGES_DICT[(n3.n_id, n1.n_id)] and EDGES_DICT[(n2.n_id, n3.n_id)] and EDGES_DICT[(n3.n_id, n2.n_id)]
        except KeyError:
            return False

    def info(self):
        return "Neurons: " + str(len(self.neurons)) + " Synapses: " + str(len(self.edges)) + " Clusters: " + str(len(self.clusters))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-k", dest="clusters", type=int)
    parser.add_argument("-n", dest="neurons", type=int)
    parser.add_argument("-m", dest="maxvalue", type=int)
    parser.add_argument("-i", dest="iterations", type=int)
    parser.add_argument("--skip-all-count", dest="skip_all", action="store_true", default=False)
    parser.add_argument("--skip-cluster-count", dest="skip_clusters", action="store_true", default=False)
    parser.add_argument("--load-brain", dest="brain_data_dir", type=str, default=None, help="timestamp dir/")
    parser.add_argument("--save-brain-nodes-and-edges", dest="persist", action="store_true", default=False)
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    parser.add_argument("--neuron-distribution", dest="neuron_distribution_type", choices=["clustered", "uniform", "random"])
    parser.add_argument("--inter-cluster-edge-prob-perct", dest="intercluster_edge_probability", type=float)
    parser.add_argument("--intra-cluster-edge-prob-perct", dest="intracluster_edge_probability", type=float)
    parser.add_argument("--inter-cluster-threshold-percentage", dest="intercluster_thresh", type=int)
    parser.add_argument("--intra-cluster-threshold-percentage", dest="intracluster_thresh", type=int)
    options = parser.parse_args()
    DEBUG = options.debug
    SAVE = options.persist
    skip_all = options.skip_all
    skip_clusters = options.skip_clusters

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
        brain.generate_neurons(options.neuron_distribution_type, options.maxvalue, clusters)
        brain.generate_synapses(intra_prob=options.intracluster_edge_probability, 
                                inter_prob=options.intercluster_edge_probability,
                                intra_thresh=options.intracluster_thresh,
                                inter_thresh=options.intercluster_thresh)
        brain.k_cluster_neurons(options.iterations)

    EDGES_DICT = brain.generate_edges_map()

    if SAVE is True:
        brain.save_brain()

    print(brain.info())

    if skip_clusters is False:
        brain.print_cluster_info()
        start_time = int(time.time())
        print("Counting cluster triangles... this may take a while")
        triangles = brain.count_cluster_triangles()
        print("Triangles: " + str(triangles))
        end_time = time.time()
        print("Runtime for cluster triangles: {0} seconds".format(str((end_time - start_time))))
        brain.save_result("Triangles: " + str(triangles))
        brain.save_result("Clusters runtime: {0}".format(str(end_time - start_time)))

    if skip_all is False:
        start_time = int(time.time())
        print("Counting ALL (~N^3 time) triangles... this may take a while")
        triangles = brain.count_all_triangles()
        print("Triangles: " + str(triangles))
        end_time = time.time()
        print("Runtime for all triangles: {0} seconds".format(str((end_time - start_time))))
        brain.save_result("Triangles: " + str(triangles))
        brain.save_result("All runtime: {0}".format(str(end_time - start_time)))




