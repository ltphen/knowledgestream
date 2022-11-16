"""
Entry point for Knowledge Stream (KS) and 
Relational Knowledge Linker (KL-REL) algorithm.
"""

import os
import argparse
import socket
import numpy as np
import logging as log

from os.path import expanduser, abspath, join, exists

from datastructures.rgraph import Graph
from algorithms.ClientService import ClientService

# OUR METHODS
from algorithms.mincostflow.ssp import disable_logging


# KG - DBpedia
HOME = abspath(expanduser('/knowledgestream/data/'))
if not exists(HOME):
    print('Knowledgegraph not found: %s' % HOME)
    print('Download Knowledgegraph per instructions on:')
    print('http://github.com/saschaTrippel/knowledgestream#knowledgegraph')
    print('and enter the directory path below.')
    data_dir = raw_input('\nPlease enter data directory path: ')
    if data_dir != '':
        data_dir = abspath(expanduser(data_dir))
    if not os.path.isdir(data_dir):
        raise Exception('Entered path "%s" not a directory.' % data_dir)
    if not exists(data_dir):
        raise Exception('Directory does not exist: %s' % data_dir)
    HOME = data_dir
PATH = join(HOME, 'kg/_undir/')
assert exists(PATH)
SHAPE = "data/kg/shape.txt"

# relational similarity using TF-IDF representation and cosine similarity
RELSIMPATH = join(HOME, 'relsim/predicate-similarity.npy') 

internalId = dict()

def parseArguments():
	# parse arguments
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-m', type=str, required=True,
			dest='method', help='Method to use: stream, relklinker, klinker, \
			predpath, pra, katz, pathent, simrank, adamic_adar, jaccard, degree_product.')
        parser.add_argument('-p', type=int, required=False, default=4444, dest='port', help='Specify on which port shall be listened.')
        return parser.parse_args()

def load_shape():
    with open(SHAPE, 'r') as shapeFile:
        line = shapeFile.readline()
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace(' ', '')
        split = line.split(',')
        return (int(split[0]), int(split[1]), int(split[2]))

def listen(connections=10, port=4444):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(connections)
    return s

def cacheIds():
    path = HOME + "/kg/"
    idFileNodes = open(path + "nodes.txt", 'r')

    for line in idFileNodes.readlines():
        intId, iri = line.split(' ')
        iri = iri.replace('\n', '')
        internalId[iri] = int(intId)

    idFileNodes.close()

    idFileRelations = open(path + "relations.txt", 'r')
    for line in idFileRelations.readlines():
        intId, iri = line.split(' ')
        iri = iri.replace('\n', '')
        internalId[iri] = int(intId)

    idFileRelations.close()


def main(args=None):
    args = parseArguments()

    # logging
    disable_logging(log.DEBUG)

    if args.method not in (
        'stream', 'relklinker', 'klinker', 'predpath', 'pra',
        'katz', 'pathent', 'simrank', 'adamic_adar', 'jaccard', 'degree_product'
        ):
        raise Exception('Invalid method specified.')

    # load knowledge graph
    shape = load_shape()
    print(shape)
    G = Graph.reconstruct(PATH, shape, sym=True) # undirected
    assert np.all(G.csr.indices >= 0)

    # relational similarity
    if args.method == 'stream' or args.method == 'relklinker':
        relsim = np.load(RELSIMPATH)
    else:
        relsim = None

    # Read internal IDs from file
    log.info('Caching internal IDs')
    cacheIds()

    # listen for connections
    s = listen(port=args.port)
    try:
        while True:
            log.info('Waiting for connection on port {}'.format(args.port))
            client, conn = s.accept()
            log.info('Accepted connection')
            clientService = ClientService(client, args.method, G, relsim, internalId)
            clientService.serve()
    except KeyboardInterrupt:
        s.close()
        print('\n')
        return

if __name__ == '__main__':
    """
    Example calls: 

    cd ~/Projects/knowledgestream/
    python setup.py develop OR python setup.py install

    # Knowledge Stream:
    kstream -m stream -p 4444

    # Relational Knowledge Linker (KL-REL)
    kstream -m 'relklinker' -p 4444

    # PredPath
    kstream -m 'predpath' -p 4444

    # PRA
    kstream -m 'pra' -p 4444
    """
    main()
