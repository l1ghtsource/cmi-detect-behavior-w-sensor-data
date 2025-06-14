from collections import namedtuple
import heapq
import os

CheckpointInfo = namedtuple('CheckpointInfo', ['score', 'epoch', 'model_path', 'ema_path'])

class TopKCheckpoints:
    def __init__(self, k=5):
        self.k = k
        self.checkpoints = []
    
    def add_checkpoint(self, score, epoch, model_path, ema_path=None):
        checkpoint = CheckpointInfo(score, epoch, model_path, ema_path)
        
        if len(self.checkpoints) < self.k:
            heapq.heappush(self.checkpoints, checkpoint)
        elif score > self.checkpoints[0].score:
            old_checkpoint = heapq.heappop(self.checkpoints)
            if os.path.exists(old_checkpoint.model_path):
                os.remove(old_checkpoint.model_path)
            if old_checkpoint.ema_path and os.path.exists(old_checkpoint.ema_path):
                os.remove(old_checkpoint.ema_path)
            
            heapq.heappush(self.checkpoints, checkpoint)
    
    def get_best_checkpoint(self):
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x.score)
    
    def cleanup_all(self):
        for checkpoint in self.checkpoints:
            if os.path.exists(checkpoint.model_path):
                os.remove(checkpoint.model_path)
            if checkpoint.ema_path and os.path.exists(checkpoint.ema_path):
                os.remove(checkpoint.ema_path)