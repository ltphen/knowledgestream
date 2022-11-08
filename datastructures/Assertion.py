class Assertion:
    def __init__(self, subjectId, predicateId, objectId):
        self.subjectId = subjectId
        self.predicateId = predicateId
        self.objectId = objectId
        self._expectedScore = None
        
    @property
    def expectedScore(self):
        return self._expectedScore
    
    @expectedScore.setter
    def expectedScore(self, score):
        if score == 0 or score == 1:
            self._expectedScore = score
