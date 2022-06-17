"""
Entry point for Knowledge Stream (KS) and 
Relational Knowledge Linker (KL-REL) algorithm.
"""

import rdflib
import sys
import os
import argparse
import socket
import numpy as np
import pandas as pd
import warnings
import ujson as json
import logging as log

from pandas import DataFrame, Series
from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from time import time
from datetime import date
import cPickle as pkl

from datastructures.rgraph import Graph, weighted_degree
from rdflib import Graph as RDFGraph

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
RELSIMPATH = join(HOME, 'relsim/coo_mat_sym_2016-10-24_log-tf_tfidf.npy') 
#assert exists(RELSIMPATH) TODO: uncomment when relsim is generated

# Date
DATE = '{}'.format(date.today())

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

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

# prefix dict
prefix = dict()
prefix['dbo'] = "http://dbpedia.org/ontology/"
prefix['dbp'] = "http://dbpedia.org/property/"
prefix['dbr'] = "http://dbpedia.org/resource/"

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

def compute_mincostflow_legacy(G, relsim, subs, preds, objs, flowfile):
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
	with open(flowfile, 'w', 0) as ff:
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
			mcflow = succ_shortest_path(
				G, cost_vec, s, p, o, return_flow=False, npaths=5
			)
			mincostflows.append(mcflow.flow)
			ff.write(json.dumps(mcflow.stream) + '\n')
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

def readData(dataset):
    df = pd.read_table(dataset, sep=',', header=0)
    log.info('Read data: {} {}'.format(df.shape, basename(dataset)))
    spo_df = df.dropna(axis=0, subset=['sid', 'pid', 'oid'])
    log.info('Note: Found non-NA records: {}'.format(spo_df.shape))
    df = spo_df[['sid', 'pid', 'oid']].values
    return df[:,0].astype(_int), df[:,1].astype(_int), df[:,2].astype(_int), spo_df

def load_shape():
    with open(SHAPE, 'r') as shapeFile:
        line = shapeFile.readline()
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace(' ', '')
        split = line.split(',')
        return (int(split[0]), int(split[1]), int(split[2]))

def ensureValidPaths(args):
    outdir = abspath(expanduser(args.outdir))
    assert exists(outdir)
    args.outdir = outdir
    datafile = abspath(expanduser(args.dataset))
    assert exists(datafile)
    args.dataset = datafile

def executeBatch(args, G, spo_df, relsim, subs, preds, objs):
	base = splitext(basename(args.dataset))[0]
	t1 = time()
	if args.method == 'stream': # KNOWLEDGE STREAM (KS)
		# compute min. cost flow
		log.info('Computing KS for {} triples..'.format(spo_df.shape[0]))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			outjson = join(args.outdir, 'out_kstream_{}_{}.json'.format(base, DATE))
			outcsv = join(args.outdir, 'out_kstream_{}_{}.csv'.format(base, DATE))
			mincostflows, times = compute_mincostflow_legacy(G, relsim, subs, preds, objs, outjson)
			# save the results
			spo_df['score'] = mincostflows
			spo_df['time'] = times
			spo_df = normalize(spo_df)
			spo_df.to_csv(outcsv, sep=',', header=True, index=False)
			log.info('* Saved results: %s' % outcsv)
		log.info('Mincostflow computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'relklinker': # RELATIONAL KNOWLEDGE LINKER (KL-REL)
		log.info('Computing KL-REL for {} triples..'.format(spo_df.shape[0]))
		scores, paths, rpaths, times = compute_relklinker(G, relsim, subs, preds, objs)
		# save the results
		spo_df['score'] = scores
		spo_df['path'] = paths
		spo_df['rpath'] = rpaths
		spo_df['time'] = times
		spo_df = normalize(spo_df)
		outcsv = join(args.outdir, 'out_relklinker_{}_{}.csv'.format(base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		log.info('* Saved results: %s' % outcsv)
		log.info('Relatioanal KL computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'klinker':
		log.info('Computing KL for {} triples..'.format(spo_df.shape[0]))
		scores, paths, rpaths, times = compute_klinker(G, subs, preds, objs)
		# save the results
		spo_df['score'] = scores
		spo_df['path'] = paths
		spo_df['rpath'] = rpaths
		spo_df['time'] = times
		spo_df = normalize(spo_df)
		outcsv = join(args.outdir, 'out_klinker_{}_{}.csv'.format(base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		log.info('* Saved results: %s' % outcsv)
		log.info('KL computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'predpath': # PREDPATH
		vec, model = predpath_train_model(G, spo_df) # train
		print 'Time taken: {:.2f}s\n'.format(time() - t1)
		# save model
		predictor = { 'dictvectorizer': vec, 'model': model }
		try:
			outpkl = join(args.outdir, 'out_predpath_{}_{}.pkl'.format(base, DATE))
			with open(outpkl, 'wb') as g:
				pkl.dump(predictor, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved: {}'.format(outpkl)
		except IOError, e:
			raise e
	elif args.method == 'pra': # PRA
		features, model = pra_train_model(G, spo_df)
		print 'Time taken: {:.2f}s\n'.format(time() - t1)
		# save model
		predictor = { 'features': features, 'model': model }
		try:
			outpkl = join(args.outdir, 'out_pra_{}_{}.pkl'.format(base, DATE))
			with open(outpkl, 'wb') as g:
				pkl.dump(predictor, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved: {}'.format(outpkl)
		except IOError, e:
			raise e
	elif args.method in ('katz', 'pathent', 'simrank', 'adamic_adar', 'jaccard', 'degree_product'):
		scores, times = link_prediction(G, subs, preds, objs, selected_measure=args.method)
		# save the results
		spo_df['score'] = scores
		spo_df['time'] = times
		spo_df = normalize(spo_df)
		outcsv = join(args.outdir, 'out_{}_{}_{}.csv'.format(args.method, base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		print '* Saved results: %s' % outcsv

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

def batch(args, G, relsim):
    # ensure input file and output directory is valid.
    ensureValidPaths(args)
    log.info('Launching {}..'.format(args.method))
    log.info('Dataset: {}'.format(basename(args.dataset)))
    log.info('Output dir: {}'.format(args.outdir))

    # read data
    subs, preds, objs, spo_df = readData(args.dataset)

    # execute
    executeBatch(args, G, spo_df, relsim, subs, preds, objs)

def listen(connections=10, port=4444):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(connections)
    return s

def parseRequest(assertionString):
    """
    Returns a RDFGraph that contains the input assertion
    """
    log.info('Parsin assertion: {}'.format(assertionString.replace('\n', '')))

    prefixString = ""
    for short, iri in prefix.items():
        prefixString += "@prefix {}: <{}> .\n".format(short, iri)

    g = RDFGraph()
    g.parse(data=prefixString + assertionString, format='ttl')
    return g

def respondToAssertion(method, rdfAssertion, graph, relsim):
    for s, p, o in rdfAssertion:
        log.info('Validating assertion "{} {} {}" using {}'.format(
            s.encode('utf-8'), p.encode('utf-8'), o.encode('utf-8'), method))
        return str(execute(method, graph, relsim, getId(s), getId(p), getId(o)))

    return "ERROR: No assertion provided."

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
        intId = internalId[abbriviate(element)]
    except KeyError as ex:
        log.info('Cannot find internal ID of {}'.format(element))
        raise ex
    return intId

def abbriviate(element):
    for short, iri in prefix.items():
        if iri in element:
            return element.replace(iri, short+":").encode('utf-8')
    return element

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
            assertion = parseRequest(request)
            response = respondToAssertion(method, assertion, graph, relsim)
            log.info('Score: {}'.format(response))
            log.info('### VALIDATION DONE ###')
            client.send(response)
        except socket.error as ex:
            log.info('Socket error occured.')
            return
        except rdflib.plugins.parsers.notation3.BadSyntax as ex:
            log.info('Exception while parsing: "{}"'.format(request))
            client.send("PARSING ERROR\n")
            continue
        except KeyError as ex:
            client.send("ID ERROR\n")
            continue
        except UnicodeEncodeError as ex:
            client.send("ENCODING ERROR\n")
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
