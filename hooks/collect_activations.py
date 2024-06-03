# Hook that saves outputs of a module. Optionally apply an activation function to the outputs.   
class CollectActivationsLinearNoMean:
    def __init__(self, unconditional=False, activation_fkt=None):
        self.unconditional=unconditional
        self.outputs = None
        self.activation_fkt = activation_fkt
        self.active = True

    def __call__(self, module, module_in, module_out):
        if self.active:
            if self.activation_fkt is not None:
                module_out = self.activation_fkt(module_out)
                                
            self.outputs = module_out.detach()
                
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False

    def clear(self):
        self.outputs = None
        
    def median_activations(self):
        if self.outputs is None:
            return None
        return self.outputs.median(0).values
        
    def activations(self):
        return self.outputs