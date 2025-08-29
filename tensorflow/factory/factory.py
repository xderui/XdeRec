import tensorflow as tf
import yaml
import importlib
from collections import defaultdict
import sys
import inspect
import copy
import json

sys.path.append('/dockerdata/aurora/single_host_model_test/code')

importlib.invalidate_caches()

class Factory(tf.keras.layers.Layer):
    def __init__(self, factory_config):
        """ class Factory

        Args:
            factory_config (dict): dict of components. {'component_1': component_1_config, 'component_2': 'component_2_config',...}

        """
        super(Factory, self).__init__()
        
        self.factory_config = factory_config
        if isinstance(factory_config, str):
            try:
                self.factory_config = yaml.load(open(factory_config, 'r'), Loader=yaml.FullLoader)
            except:
                raise ValueError("请检查factory配置文件路径以及格式")
            
        assert isinstance(self.factory_config, dict), "factory配置的数据类型需为dict"
        
        self.register()
        
        
    def _check_graph_valid_(self):
        # check start
        assert self.startName in self.graphNext, "No start"
        
        # check self-cycle
        flag = defaultdict(lambda : 0) # 0: 未遍历 1: 遍历中 2: 已遍历
        cycleRoute = []
        def existCycle(node, records):
            if flag[node] == 1:
                return True
            if flag[node] == 2:
                return False
            
            records.append(node)
            flag[node] = 1
            haveCycle = False
            for nextNode in self.graphNext[node]:
                haveCycle_ = existCycle(nextNode, records)
                haveCycle = haveCycle or haveCycle_
                if haveCycle:
                    if flag[nextNode] == 1:
                        nonlocal cycleRoute
                        cycleRoute.append(records + [nextNode])
            
            records.pop()
            flag[node] = 2
            return haveCycle, records
        
        _vis_flag_ = False
        for node in self.graphNext[self.startName]:
            if flag[node] == 0:
                if _vis_flag_:
                    raise ValueError("It's not a full-connection graph.")
                records = []
                haveCycle, records = existCycle(node, records)
                _vis_flag_ = True
                if haveCycle:
                    raise ValueError("There are cycle in the graph. {}".format(" | ".join(["->".join(cycle_route) for cycle_route in cycleRoute])))
        
        
    def _construct_execution_layers_(self):
        # bfs
        pass
            
    
    @staticmethod
    def find_all_classes(module):
        return [member[1] for member in inspect.getmembers(module, inspect.isclass)
                if member[1].__module__ == module.__name__]
        
        
    def register(self):
        self.components = defaultdict(None)
        self.graphNext = defaultdict(list)
        self.startName = self.factory_config['start_name']
        self.components[self.startName] = None
        

        for component_key, component_config in self.factory_config["components"].items():
            component_class = "_".join(component_key.split('_')[:-1])
            component_module = importlib.import_module(component_class)
            print(self.find_all_classes(component_module)[0])
            componentInstance = self.find_all_classes(component_module)[0](component_config)
            self.components[component_key] = componentInstance
            
        for link in self.factory_config['graph']:
            component_num = len(link)
            assert link[0] in self.components, f"{link[0]} has no definition!"
            for index in range(component_num-1):
                current_component_key = link[index]
                next_component_key = link[index+1]
                assert next_component_key in self.components, f"{next_component_key} has no definition!"
                self.graphNext[current_component_key].append(next_component_key)

        self._check_graph_valid_()
        self._construct_execution_layers_()
        
    def __len__(self):
        return len(self.components)
    
    def __str__(self):
        return "============ Factory ============\n" + \
            "\n".join(["{}:{}".format(key, value) for key, value in self.components.items()])
            
    def output_graph(self):
        pass
    
    def run(self):
        # 串行执行节点
        # inputs =
        pass

if __name__ == "__main__":
    factory_config = "./config/model/hstu.factory.yaml"
    factory = Factory(factory_config)
    print(factory)
    print(len(factory))
    
    
    
    
    