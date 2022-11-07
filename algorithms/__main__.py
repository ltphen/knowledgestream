"""
Entry point for Knowledge Stream (KS) and 
Relational Knowledge Linker (KL-REL) algorithm.
"""

import sys
import os
import argparse
import socket
import numpy as np
import warnings
import logging as log

from os.path import expanduser, abspath, join, exists
from time import time

from datastructures.rgraph import Graph, weighted_degree
from datastructures.Assertion import Assertion
from datastructures.Message import Message

# OUR METHODS
from algorithms.mincostflow.ssp import succ_shortest_path, disable_logging
from algorithms.relklinker.rel_closure import relational_closure as relclosure
from algorithms.klinker.closure import closure

# STATE-OF-THE-ART ALGORITHMS
from algorithms.predpath.predpath_mining import train_model as predpath_train_model
from algorithms.pra.pra_mining import train_model as pra_train_model
from algorithms.linkpred.katz import katz
from algorithms.linkpred.pathentropy import pathentropy
from algorithms.linkpred.simrank import c_simrank
from algorithms.linkpred.jaccard_coeff import jaccard_coeff
from algorithms.linkpred.adamic_adar import adamic_adar
from algorithms.linkpred.pref_attach import preferential_attachment


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
WTFN = 'logdegree'

# relational similarity using TF-IDF representation and cosine similarity
RELSIMPATH = join(HOME, 'relsim/predicate-similarity.npy') 
# assert exists(RELSIMPATH)

# link prediction measures
measure_map = {
	'jaccard': {
		'measure': jaccard_coeff,
		'tag': 'JC'
	},
	'adamic_adar': {
		'measure': adamic_adar,
		'tag': 'AA'
	},
	'degree_product': {
		'measure': preferential_attachment,
		'tag': 'PA'
	},
	'katz': {
		'measure': katz,
		'tag': 'KZ'
	},
	'simrank': {
		'measure': c_simrank,
		'tag': 'SR'
	},
	'pathent': {
		'measure': pathentropy,
		'tag': 'PE'
	}
}

internalId = dict()


# ================= KNOWLEDGE STREAM ALGORITHM ============

def compute_mincostflow(G, relsim, subs, preds, objs):
    """
    Parameters:
    -----------
    G: rgraph
            See `datastructures`.
    relsim: ndarray
            A square matrix containing relational similarity scores.
    subs, preds, objs: sequence
            Sequences representing the subject, predicate and object of 
            input triples.
    flowfile: str
            Absolute path of the file where flow will be stored as JSON,
            one line per triple.

    Returns:
    --------
    mincostflows: sequence
            A sequence containing total flow for each triple.
    times: sequence
            Times taken to compute stream of each triple. 
    """
    # take graph backup
    G_bak = {
	'data': G.csr.data.copy(), 
	'indices': G.csr.indices.copy(),
	'indptr': G.csr.indptr.copy()
    }
    cost_vec_bak = np.log(G.indeg_vec).copy()

    # some set up
    G.sources = np.repeat(np.arange(G.N), np.diff(G.csr.indptr))
    G.targets = G.csr.indices % G.N
    cost_vec = cost_vec_bak.copy()
    indegsim = weighted_degree(G.indeg_vec, weight=WTFN)
    specificity_wt = indegsim[G.targets] # specificity
    relations = (G.csr.indices - G.targets) / G.N
    mincostflows, times = [], []
    for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
        s, p, o = [int(x) for x in (s, p, o)]
        ts = time()
        if len(subs) > 1:
            print '{}. Working on {} .. '.format(idx+1, (s, p, o)),
            sys.stdout.flush()

        # set weights
        relsimvec = np.array(relsim[p, :]) # specific to predicate p
        relsim_wt = relsimvec[relations]
        G.csr.data = np.multiply(relsim_wt, specificity_wt)
			
        # compute
        mcflow = succ_shortest_path(G, cost_vec, s, p, o, return_flow=False, npaths=5)
	mincostflows.append(mcflow.flow)
        tend = time()
        times.append(tend - ts)
        if len(subs) > 1:
            print 'mincostflow: {:.5f}, #paths: {}, time: {:.2f}s.'.format(
	        mcflow.flow, len(mcflow.stream['paths']), tend - ts)

        # reset state of the graph
        np.copyto(G.csr.data, G_bak['data'])
        np.copyto(G.csr.indices, G_bak['indices'])
        np.copyto(G.csr.indptr, G_bak['indptr'])
        np.copyto(cost_vec, cost_vec_bak)
	return mincostflows, times

# ================= RELATIONAL KNOWLEDGE LINKER ALGORITHM ============

def compute_relklinker(G, relsim, subs, preds, objs):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	relsim: ndarray
		A square matrix containing relational similarity scores.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of 
		input triples.

	Returns:
	--------
	scores, paths, rpaths, times: sequence
		One sequence each for the proximity scores, shortest path in terms of 
		nodes, shortest path in terms of relation sequence, and times taken.
	"""
	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# relation vector
	relations = (G.csr.indices - targets) / G.N

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	scores, paths, rpaths, times = [], [], [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
                if len(subs) > 1:
		    print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		ts = time()
		# set relational weight
		G.csr.data[targets == o] = 1 # no cost for target t => max. specificity.
		relsimvec = relsim[p, :] # specific to predicate p
		relsim_wt = relsimvec[relations] # graph weight
		G.csr.data = np.multiply(relsim_wt, G.csr.data)

		rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
		tend = time()
		print 'time: {:.2f}s'.format(tend - ts)
		times.append(tend - ts)
		scores.append(rp.score)
		paths.append(rp.path)
		rpaths.append(rp.relational_path)

		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	return scores, paths, rpaths, times

# ================= KNOWLEDGE LINKER ALGORITHM ============

def compute_klinker(G, subs, preds, objs):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of 
		input triples.

	Returns:
	--------
	scores, paths, rpaths, times: sequence
		One sequence each for the proximity scores, shortest path in terms of 
		nodes, shortest path in terms of relation sequence, and times taken.
	"""
	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	# compute closure
	scores, paths, rpaths, times = [], [], [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
                if len(subs) > 1:
		    print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		ts = time()
		rp = closure(G, s, p, o, kind='metric', linkpred=True)
		tend = time()
                if len(subs) > 1:
		    print 'time: {:.2f}s'.format(tend - ts)
		times.append(tend - ts)
		scores.append(rp.score)
		paths.append(rp.path)
		rpaths.append(rp.relational_path)

		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	return scores, paths, rpaths, times

def normalize(df):
	softmax = lambda x: np.exp(x) / float(np.exp(x).sum())
	df['softmaxscore'] = df[['sid','score']].groupby(by=['sid'], as_index=False).transform(softmax)
	return df


# ================= LINK PREDICTION ALGORITHMS ============

def link_prediction(G, subs, preds, objs, selected_measure='katz'):
	"""
	Performs link prediction using a specified measure, such as Katz or SimRank.

	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of 
		input triples.

	Returns:
	--------
	scores, times: sequence
		One sequence each for the proximity scores and times taken.
	"""
	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	# compute closure
	measure_name = measure_map[selected_measure]['tag']
	measure = measure_map[selected_measure]['measure']
	log.info('Computing {} for {} triples..'.format(measure_name, len(subs)))
	t1 = time()
	scores, times = [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
                if len(subs) > 1:
		    print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		sys.stdout.flush()
		ts = time()
		score = measure(G, s, p, o, linkpred=True)
		tend = time()
                if len(subs) > 1:
                    print 'score: {:.5f}, time: {:.2f}s'.format(score, tend - ts)
		times.append(tend - ts)
		scores.append(score)

		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	print ''
	return scores, times

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
    t1 = time()
    if method == 'stream': # KNOWLEDGE STREAM (KS)
        with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    mincostflows, times = compute_mincostflow(G, relsim, [subId], [predId], [objId])
            return mincostflows[0]
    elif method == 'relklinker': # RELATIONAL KNOWLEDGE LINKER (KL-REL)
        scores, paths, rpaths, times = compute_relklinker(G, relsim, [subId], [predId], [objId])
        return scores[0]
    elif method == 'klinker':
        scores, paths, rpaths, times = compute_klinker(G, [subId], [predId], [objId])
        return scores[0]
    elif method == 'predpath': # PREDPATH
        # TODO: this
        vec, model = predpath_train_model(G, spo_df) # train
    elif method == 'pra': # PRA
        # TODO: this
        features, model = pra_train_model(G, spo_df)
    elif method in ('katz', 'pathent', 'simrank', 'adamic_adar', 'jaccard', 'degree_product'):
        scores, times = link_prediction(G, [subId], [predId], [objId], selected_measure=method)
        return scores[0]

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
