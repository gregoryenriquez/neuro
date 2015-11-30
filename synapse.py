class Synapse:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def setStart(self, start):
        self.start = start

    def setEnd(self, end):
        self.end = end

    def info(self):
        return "S:\t{0}\tE:\t{1}".format(self.start, self.end)

