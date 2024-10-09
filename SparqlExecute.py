import rdflib

# Read the nt file to generate knowledge graph
graph = rdflib.Graph()
graph.parse('Dataset/14_graph.nt', format='turtle')


def query(sparql):
    return str([str(s) for s, in graph.query(sparql)])
