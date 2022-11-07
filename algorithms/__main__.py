"""
Entry point for Knowledge Stream (KS) and 
Relational Knowledge Linker (KL-REL) algorithm.
"""

import os
import argparse
import socket
import numpy as np
import warnings
import logging as log

from os.path import expanduser, abspath, join, exists
from time import time

from datastructures.rgraph import Graph
from datastructures.Assertion import Assertion
from datastructures.Message import Message
from algorithms.AlgorithmRunner import AlgorithmRunner

# OUR METHODS
from algorithms.mincostflow.ssp import disable_logging

# STATE-OF-THE-ART ALGORITHMS
from algorithms.predpath.predpath_mining import train_model as predpath_train_model
from algorithms.pra.pra_mining import train_model as pra_train_model


# KG - DBpedia
HOME = abspath(expanduser('/knowledgestream/data/'))
if not exists(HOME):
    print 'Data directory not found: %s' % HOME
    print 'Download data per instructions on:'
    print '\thttps://github.com/shiralkarprashant/knowledgestream#data'
    print 'and enter the directory path below.'
    data_dir = raw_input('\nPlease enter data directory path: ')
    if data_dir != '':
	data_dir = abspath(expanduser(data_dir))
    if not os.path.isdir(data_dir):
	raise Exception('Entered path "%s" not a directory.' % data_dir)
    if not exists(data_dir):
	raise Exception('Directory does not exist: %s' % data_dir)
    HOME = data_dir
    # raise Exception('Please set HOME to data directory in algorithms/__main__.py')
PATH = join(HOME, 'kg/_undir/')
assert exists(PATH)
SHAPE = "data/kg/shape.txt"

# relational similarity using TF-IDF representation and cosine similarity
RELSIMPATH = join(HOME, 'relsim/predicate-similarity.npy') 
# assert exists(RELSIMPATH)

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
	parser.add_argument('-d', type=str, required=False,
			dest='dataset', help='Dataset to test on.')
	parser.add_argument('-o', type=str, required=False,
			dest='outdir', help='Path to the output directory.')
        parser.add_argument('-b', type=bool, required=False, default=False, dest='batch', help='Run in batch mode and read input from file.')
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

def execute(method, G, relsim, subId, predId, objId):
    """
    Validate a single assertion.
    """
    algo = AlgorithmRunner(method, G, relsim)
    return algo.validate(subId, predId, objId)

def listen(connections=10, port=4444):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(connections)
    return s

def parseRequest(assertionString):
    log.info('Parsin assertion: {}'.format(assertionString.replace('\n', '')))
    return Message(text=assertionString)

def respondToRequest(method, request, graph, relsim):
    if request.type == "call" and request.content == "type":
        if method in ["predpath", "pra"]:
            return Message(type="type_response", content="supervised")
        else:
            return Message(type="type_response", content="unsupervised")

    if request.type == "test":
        log.info('Validating assertion "{} {} {}" using {}'.format(
            request.subject.encode('utf-8'), request.predicate.encode('utf-8'), request.object.encode('utf-8'), method))
        result = execute(method, graph, relsim, getId(request.subject), getId(request.predicate), getId(request.object))
        return Message(type="test_result", score="{:f}".format(result))

    return Message(type="error", content="Something went wrong.")

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

def getId(element):
    try:
        intId = internalId[str(element.encode('utf-8'))]
    except KeyError as ex:
        log.info('Cannot find internal ID of {}'.format(element.encode('utf-8')))
        raise ex
    return intId

def serviceClient(method, client, graph, relsim):
    while True:
        try:
            log.info('Waiting for an assertion')
            request = client.recv(1024)
            if request == '':
                log.info('Connection closed')
                client.close()
                return
            log.info('### VALIDATION START ###')
            requestMessage = parseRequest(request)
            response = respondToRequest(method, requestMessage, graph, relsim)
            log.info('### VALIDATION DONE ###')
            client.send(response.serialize())
        except socket.error as ex:
            log.info('Socket error occured.')
            return
        except KeyError as ex:
            client.send(Message(type="error", content="ID Error").serialize())
            continue
        except UnicodeEncodeError as ex:
            client.send(Message(type="error", content="Encoding Error").serialize())
            continue
        except Exception as ex:
            raise ex

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

    if (args.batch):
        batch(args, G, relsim)
        print '\nDone!\n'
        return

    # Read internal IDs from file
    log.info('Caching internal IDs')
    cacheIds()

    # listen for connections
    print
    s = listen(port=args.port)
    try:
        while True:
            log.info('Waiting for connection on port {}'.format(args.port))
            client, conn = s.accept()
            log.info('Accepted connection')
            serviceClient(args.method, client, G, relsim)
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
    kstream -m 'stream' -d ./datasets/synthetic/Player_vs_Team_NBA.csv -o ./output/
    kstream -m 'stream' -d ./datasets/sample.csv -o ./output/

    # Relational Knowledge Linker (KL-REL)
    kstream -m 'relklinker' -d ./datasets/sample.csv -o ./output/

    # PredPath
    kstream -m 'predpath' -d ./datasets/sample.csv -o ./output/	

    # PRA
    kstream -m 'pra' -d ./datasets/sample.csv -o ./output/	
    """
    main()
