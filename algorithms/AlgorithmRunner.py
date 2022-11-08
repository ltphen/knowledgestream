# OUR METHODS
from algorithms.mincostflow.ssp import succ_shortest_path
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

from datastructures.rgraph import weighted_degree
from datastructures.Assertion import Assertion
from time import time
import numpy as np
import pandas as pd
import logging as log
import warnings
import sys


WTFN = 'logdegree'


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

class AlgorithmRunner:

    def __init__(self, method, G, internalId, relsim=None):
        self.method = method
        self.G = G
        # relsim only requred for KS and KL-REL
        self.relsim = relsim
        self.internalId = internalId
        self.trainingData = []
        
    def test(self, sub, pred, obj):
        log.info('Validating assertion "{} {} {}" using {}'.format(
            sub.encode('utf-8'), pred.encode('utf-8'), obj.encode('utf-8'), self.method))
        return self.validate(self.getId(sub), self.getId(pred), self.getId(obj))

    def validate(self, subId, predId, objId):
        """
        Validate a single assertion.
        """
        if self.method == 'stream': # KNOWLEDGE STREAM (KS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mincostflows, times = self.compute_mincostflow(self.G, self.relsim, [subId], [predId], [objId])
            return mincostflows[0]
        elif self.method == 'relklinker': # RELATIONAL KNOWLEDGE LINKER (KL-REL)
            scores, paths, rpaths, times = self.compute_relklinker(self.G, self.relsim, [subId], [predId], [objId])
            return scores[0]
        elif self.method == 'klinker':
            scores, paths, rpaths, times = self.compute_klinker(self.G, [subId], [predId], [objId])
            return scores[0]
        elif self.method == 'predpath': # PREDPATH
            # TODO: this
            vec, model = predpath_train_model(self.G, spo_df) # train
        elif self.method == 'pra': # PRA
            # TODO: this
            features, model = pra_train_model(self.G, spo_df)
        elif self.method in ('katz', 'pathent', 'simrank', 'adamic_adar', 'jaccard', 'degree_product'):
            scores, times = self.link_prediction(self.G, [subId], [predId], [objId], selected_measure=self.method)
            return scores[0]

    def addTrainingData(self, sub, pred, obj, expectedScore):
        assertion = Assertion(self.getId(sub), self.getId(pred), self.getId(obj))
        assertion.expectedScore = expectedScore
        self.trainingData.append(assertion)

    def train(self):
        trainingDf = self._createTrainingDataFrame()

        if self.method == 'predpath': # PREDPATH
            # TODO: this
            self.vec, self.model = predpath_train_model(self.G, trainingDf)
        elif self.method == 'pra': # PRA
            # TODO: this
            features, model = pra_train_model(self.G, trainingDf)

    def getId(self, element):
        try:
            intId = self.internalId[str(element.encode('utf-8'))]
        except KeyError as ex:
            log.info('Cannot find internal ID of {}'.format(element.encode('utf-8')))
            raise ex
        return intId

    def _createTrainingDataFrame(self):
        dictList = []
        for assertion in self.trainingData:
            tmpDict = dict()
            tmpDict["sid"] = assertion.subjectId
            tmpDict["pid"] = assertion.predicateId
            tmpDict["oid"] = assertion.objectId
            tmpDict["class"] = assertion.expectedScore
            dictList.append(tmpDict)

        return pd.DataFrame(dictList)


    
    # ================= KNOWLEDGE STREAM ALGORITHM ============
    
    def compute_mincostflow(self, G, relsim, subs, preds, objs):
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
                print ('{}. Working on {} .. '.format(idx+1, (s, p, o)))
    
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
                print ('mincostflow: {:.5f}, #paths: {}, time: {:.2f}s.'.format(mcflow.flow, len(mcflow.stream['paths']), tend - ts))
    
            # reset state of the graph
            np.copyto(G.csr.data, G_bak['data'])
            np.copyto(G.csr.indices, G_bak['indices'])
            np.copyto(G.csr.indptr, G_bak['indptr'])
            np.copyto(cost_vec, cost_vec_bak)
        return mincostflows, times
    
    # ================= RELATIONAL KNOWLEDGE LINKER ALGORITHM ============
    
    def compute_relklinker(self, G, relsim, subs, preds, objs):
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
                print ('{}. Working on {}..'.format(idx+1, (s, p, o)))
            ts = time()
            # set relational weight
            G.csr.data[targets == o] = 1 # no cost for target t => max. specificity.
            relsimvec = relsim[p, :] # specific to predicate p
            relsim_wt = relsimvec[relations] # graph weight
            G.csr.data = np.multiply(relsim_wt, G.csr.data)
    
            rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
            tend = time()
            print('time: {:.2f}s'.format(tend - ts))
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
    
    def compute_klinker(self, G, subs, preds, objs):
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
                print('{}. Working on {}..'.format(idx+1, (s, p, o)))
            ts = time()
            rp = closure(G, s, p, o, kind='metric', linkpred=True)
            tend = time()
            if len(subs) > 1:
                print('time: {:.2f}s'.format(tend - ts))
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
    
    def normalize(self, df):
        softmax = lambda x: np.exp(x) / float(np.exp(x).sum())
        df['softmaxscore'] = df[['sid','score']].groupby(by=['sid'], as_index=False).transform(softmax)
        return df
    
    
    # ================= LINK PREDICTION ALGORITHMS ============
    
    def link_prediction(self, G, subs, preds, objs, selected_measure='katz'):
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
                print('{}. Working on {}..'.format(idx+1, (s, p, o)))
                sys.stdout.flush()
            ts = time()
            score = measure(G, s, p, o, linkpred=True)
            tend = time()
            if len(subs) > 1:
                print('score: {:.5f}, time: {:.2f}s'.format(score, tend - ts))
            times.append(tend - ts)
            scores.append(score)
    
            # reset graph
            G.csr.data = data.copy()
            G.csr.indices = indices.copy()
            G.csr.indptr = indptr.copy()
            sys.stdout.flush()
        print('')
        return scores, times
