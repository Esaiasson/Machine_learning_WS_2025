class Node: 
    def __init__(self, attribute, split_value, mean):
        self.attribute = attribute
        self.split_value = split_value
        self.mean = mean
        self.left = None
        self.right = None
        
    def set_left(self, node):
        self.left=node   
        
    def set_right(self, node):
        self.right=node  
            
