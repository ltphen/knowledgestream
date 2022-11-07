import logging as log
import socket
from datastructures.Message import Message
from algorithms.AlgorithmRunner import AlgorithmRunner

class ClientService:

    def __init__(self, client, method, G, relsim, internalId):
        self.client = client
        self.internalId = internalId
        self.method = method
        self.G = G
        self.relsim = relsim
        self.algoRunner = AlgorithmRunner(method, G, relsim)

    def serve(self):
        while True:
            try:
                log.info('Waiting for an assertion')
                request = self.client.recv(1024)
                if request == '':
                    log.info('Connection closed')
                    self.client.close()
                    return
                log.info('### VALIDATION START ###')
                requestMessage = self.parseRequest(request)
                response = self.respondToRequest(self.method, requestMessage, self.G, self.relsim)
                log.info('### VALIDATION DONE ###')
                self.client.send(response.serialize())
            except socket.error as ex:
                log.info('Socket error occured.')
                return
            except KeyError as ex:
                self.client.send(Message(type="error", content="ID Error").serialize())
                continue
            except UnicodeEncodeError as ex:
                self.client.send(Message(type="error", content="Encoding Error").serialize())
                continue
            except Exception as ex:
                raise ex

    def execute(self, method, G, relsim, subId, predId, objId):
        """
        Validate a single assertion.
        """
        algo = AlgorithmRunner(method, G, relsim)
        return algo.validate(subId, predId, objId)
    
    def parseRequest(self, assertionString):
        log.info('Parsin assertion: {}'.format(assertionString.replace('\n', '')))
        return Message(text=assertionString)
    
    def respondToRequest(self, method, request, graph, relsim):
        if request.type == "call" and request.content == "type":
            if method in ["predpath", "pra"]:
                return Message(type="type_response", content="supervised")
            else:
                return Message(type="type_response", content="unsupervised")
    
        if request.type == "test":
            log.info('Validating assertion "{} {} {}" using {}'.format(
                request.subject.encode('utf-8'), request.predicate.encode('utf-8'), request.object.encode('utf-8'), method))
            result = self.execute(method, graph, relsim, self.getId(request.subject), self.getId(request.predicate), self.getId(request.object))
            return Message(type="test_result", score="{:f}".format(result))
    
        return Message(type="error", content="Something went wrong.")


    def getId(self, element):
        try:
            intId = self.internalId[str(element.encode('utf-8'))]
        except KeyError as ex:
            log.info('Cannot find internal ID of {}'.format(element.encode('utf-8')))
            raise ex
        return intId
