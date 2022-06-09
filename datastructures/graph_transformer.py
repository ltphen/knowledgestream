from rdflib import Graph as RdfGraph

class GraphTransformer:
    """
    Transform a graph in turtle representation into adjacency matrix
    requred to build Graph.
    """

    def __init__(self):
        self.id = dict()

    def generateAdjacency(self, graphPath):
        rdfGraph = self._readTurtleGraph(graphPath)
        self._generateIndices(rdfGraph)


    def _generateIndices(self, rdfGraph):
        pass

    def _readTurtleGraph(self, graphPath):
        rdfGraph = RdfGraph()
        rdfGraph.parse(graphPath, format='ttl')
        return rdfGraph

g = GraphTransformer()
g.generateAdjacency("/home/sascha/dbpedia.ttl")
