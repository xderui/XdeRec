import tensorflow as tf
from typing_extensions import Dict, Any, List
import inspect
from collections import defaultdict

class Evaluator:
    def __init__(self, ):
        self.function_dict = self.get_function_dict()
    
    def hit(self, logits, labels, topk_list=[10]):
        '''
            logits: [b * n, num_classes]
            labels: [b * n, 1]
        '''
        
        max_k = max(topk_list)
        preds = tf.nn.top_k(logits, k=max_k).indices
        correct_mask = tf.cast(preds == tf.tile(labels, [1, tf.shape(preds)[-1]]), tf.int32)
        
        topk_dict = {}
        for topk in topk_list:
            topk_dict[topk] = float(tf.reduce_sum(correct_mask[:, :topk])) / preds.shape[0]
        
        return topk_dict
        
    def get_function_dict(self):
        return dict(inspect.getmembers(self.__class__, predicate=inspect.isfunction))
    
    def eval(self, logits, labels, eval_config : Dict[str, List]):
        '''
            eval_config: {
                "hit": [10,20,30...]
                ......
            }
        '''
        metric_dict = defaultdict(lambda : defaultdict(float))
        for eval_metric, topk_list in eval_config.items():
            metric_dict[eval_metric] = self.function_dict[eval_metric](self, logits, labels, topk_list)
                
        return metric_dict
        
            
if __name__ == '__main__':
    Evaluator_ = Evaluator()
    print(Evaluator_.get_function_dict())
    print(Evaluator.__name__, Evaluator_.__class__)
    print(Evaluator_.hit(tf.Variable([[4,8,7],[1,2,6]]), tf.Variable([[2],[0]]), topk=2))
    
    print(Evaluator_.eval(tf.Variable([[4,8,7],[1,2,6]]), tf.Variable([[2],[0]]), {"hit":[1,2]}))