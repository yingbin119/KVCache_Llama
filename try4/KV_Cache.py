import logging

class RecordStorage:
    def __init__(self, max_size):
        self.max_size = max_size          # 缓存容量上限
        self.storage = []                 # 用于存储最近的记录（FIFO）
        self.all_keywords = set()         # 收集所有出现过的关键词

    def _to_list_if_tensor(self, x):
            if x is None:
                return None
            try:
                # 兼容 torch tensor：detach->cpu->tolist
                if hasattr(x, "detach") and hasattr(x, "cpu"):
                    return x.detach().cpu().tolist()
                if hasattr(x, "cpu") and hasattr(x, "tolist"):
                    return x.cpu().tolist()
                if hasattr(x, "tolist"):
                    return x.tolist()
            except Exception:
                pass
            return x
    
    def add(self, keywords=None, inputs_id=None, layer_id=None, K_states=None, V_states=None):
        """添加或补充一条数据"""
        # K_states = self._to_list_if_tensor(K_states)
        # V_states = self._to_list_if_tensor(V_states)
        # inputs_id = inputs_id.squeeze()
        # inputs_id = self._to_list_if_tensor(inputs_id)
        # 情况 1：只传 keywords（先存关键词）
        if keywords is not None and K_states is None and V_states is None:
            inputs_id = inputs_id.squeeze()
            inputs_id = self._to_list_if_tensor(inputs_id)
            record = {"keywords": keywords, 
                      "inputs_id":inputs_id, 
                      "states": []}
            if len(self.storage) >= self.max_size:
                removed = self.storage.pop(0)
                logging.info("add调用, 容量已满, 移除最早的数据: %s",removed['keywords'])
            self.storage.append(record)
            logging.info("add调用, 已保存关键词, 等待后续 states, 当前缓存数量: %s ",len(self.storage))
        
        # 情况 2：只传 K_states 或 V_states（更新最后一条）
        elif (K_states is not None or V_states is not None) and keywords is None:
            if not self.storage:
                logging.info("当前没有可补充 states 的记录。请先添加关键词。")
                return
            last_record = self.storage[-1]
            last_record["states"].append({
                "layer_id": layer_id,
                "K_states": K_states,
                "V_states": V_states
            })
            logging.info("已补充layer_id: %s , K_states: %s 和 V_states: %s 到最近一条记录。",layer_id, K_states.shape, V_states.shape)        
        # 情况 3：同时传入 keywords 和 states（一次性完整添加）
        elif keywords is not None and (K_states is not None or V_states is not None):
            record = {
                "keywords": keywords,
                "inputs_id": inputs_id,
                "states": [
                    {"layer_id": layer_id, 
                     "K_states": K_states, 
                     "V_states": V_states}
                ]
            }
            if len(self.storage) >= self.max_size:
                removed = self.storage.pop(0)
                logging.info("容量已满, 移除最早的数据: %s",removed['keywords'])
            self.storage.append(record)
            logging.info(f"已完整保存记录（当前数量: {len(self.storage)}）")
        
        else:
            logging.info("请至少传入 keywords 或 states (K_states/V_states) 之一。")

    def check(self):
        keywords_list = [record['keywords'] for record in self.storage]
        inputs_id_list = [record['inputs_id'] for record in self.storage]
        return keywords_list, inputs_id_list
    
    def show(self):
        """查看当前缓存（只展示关键词）"""
        logging.info("show 调用, 当前缓存数量: %s",len(self.storage))
        keywords_list = [record['keywords'] for record in self.storage]
        inputs_id_list = [record['inputs_id'] for record in self.storage]
        for i, kws in enumerate(keywords_list, 1):
            logging.info("show 调用, 序号: %s, 关键词: %s",i,kws)
        for i, inputs_id in enumerate(inputs_id_list, 1):
            logging.info("show 调用, 序号: %s, 关键词id: %s",i,inputs_id)
        return None

    def clear(self):
        """清空缓存"""
        self.storage.clear()
        logging.info("缓存已清空")
