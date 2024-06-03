# Hook that scales the activations of a linear layer by a factor.
class RescaleLinearActivations:
    def __init__(self, indices=[], factor=0.0):
        self.indices = indices
        self.factor = factor
        self.active = True
        
    def __call__(self, module, module_in, module_out):
        if self.active == False:
            return module_out
        
        if len(self.indices) == 0:
            return module_out
        
        module_out[:, :, self.indices] *= self.factor
        return module_out
    
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False

    def set_indices(self, indices):
        self.indices = indices
                