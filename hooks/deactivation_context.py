class DeactivateHooksContext:
    '''
    Context manager to deactivate hooks temporarily.
    '''
    def __init__(self, hooks):
        '''
        Args:
            hooks: List or dict of hooks to deactivate.
        '''
        self.hooks = hooks

    def __enter__(self):
        if isinstance(self.hooks, dict):
            for hook in self.hooks.values():
                hook.deactivate()
        elif isinstance(self.hooks, list):
            for hook in self.hooks:
                hook.deactivate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.hooks, dict):
            for hook in self.hooks.values():
                hook.activate()
        elif isinstance(self.hooks, list):
            for hook in self.hooks:
                hook.activate()
