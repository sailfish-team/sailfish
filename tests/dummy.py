class DummyLogger(object):
    def debug(*args):
        pass

    def info(*args):
        pass

class DummyEvent(object):
    def is_set(self):
        return False
