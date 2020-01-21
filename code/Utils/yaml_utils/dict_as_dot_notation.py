class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
                    if isinstance(v, dict):
                        self[k] = Map(v)

        if kwargs:
            # for python 3 use kwargs.items()
            for k, v in kwargs.items():
                self[k] = v
                if isinstance(v, dict):
                    self[k] = Map(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):  # required for successful pickling
        return self.__dict__

    def __setstate__(self, d):  # required for successful pickling
        self.__dict__.update(d)


# Usage example:
# 
# from Utils.dict_as_dot_notation import Map
# m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
# # Add new key
# m.new_key = 'Hello world!'
# # Or
# m['new_key'] = 'Hello world!'
# print m.new_key
# print m['new_key']
# # Update values
# m.new_key = 'Yay!'
# # Or
# m['new_key'] = 'Yay!'
# # Delete key
# del m.new_key
# # Or
# del m['new_key']
#
#
