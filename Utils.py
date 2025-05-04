
class DynamicAttributes:
    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name):
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"