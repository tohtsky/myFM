class DefaultMapper(dict):
    def __init__(self, items):
        super().__init__()
        unique_items = list(set(items))
        super().update({ item: i+1 for i, item in enumerate(unique_items)})
        self._items = unique_items
    
    def __getitem__(self, x):
        return self.get(x, 0)
    
    def names(self):
        return ['<UNK>'] + self._items
    
    def __len__(self):
        return super().__len__() + 1
        