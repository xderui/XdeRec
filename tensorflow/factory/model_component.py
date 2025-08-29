import tensorflow as tf
import importlib
from typing import Dict, Any
import sys


# please keep the class at the top posiiton, thx
class ModelComponent(tf.keras.layers.Layer):
    """ Node for model

    Args:
        tf (_type_): _description_
        config (dict): {
                        "model_name": str,
                        "model_param": dict
                    }
    """
    def __init__(self, model_config):
        super(ModelComponent, self).__init__()
        
        self.model_config = model_config
        self.model_module, self.model_class = self.model_config.get("name", "basemodel.baseModel").split('.')
        self.model_param = self.model_config.get("param", {})
        self.model_input_sepc = self.model_config.get("input_spec", {})
        
        print(self.model_module, self.model_class)
        print(sys.path)
        MODEL_MODULE = importlib.import_module(f"model.{self.model_module}")
        MODEL_CLASS = getattr(MODEL_MODULE, self.model_class)
        
        self.model = MODEL_CLASS(**self.model_param)
        
        
        # 添加输入验证到call
        print(self.model_input_sepc)
        self.call = self.validate_input_spec(self.model_input_sepc)(self.call)
    
        
    # 输入验证装饰器
    def validate_input_spec(self, input_spec_config):
        def decorator(f):
            def wrapper(self, inputs):
                for input_spec_ in input_spec_config:  # input_spec_ : dict
                    assert input_key in inputs, f"Missing key {input_key} in inputs"  # check input
                    assert isinstance(inputs[input_key], type_), f"Expect {type_} for {input_key}"
                    assert len(tf.shape(inputs[input_key])) == len(shape_), "The number of dimensions are not matched"  
                    name_ = input_spec_['name']
                    type_ = input_spec_['type']
                    shape_ = input_spec_['shape']
                                      
                return f(self, inputs)
        return decorator

    def call(self, inputs: Dict[str, Any]):
        return self.model(**inputs)
    
class Test:
    NotImplemented
    