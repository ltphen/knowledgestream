class Assertion:
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self._expectedScore = None
        self.score = None
        
    @property
    def expectedScore(self):
        return self._expectedScore
    
    @expectedScore.setter
    def expectedScore(self, score):
        if score == 0 or score == 1:
            self._expectedScore = score
