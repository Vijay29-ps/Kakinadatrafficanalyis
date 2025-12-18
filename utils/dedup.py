import time

class EventDeduplicator:
    def __init__(self, cooldown=5):
        self.cooldown = cooldown
        self.cache = {}

    def allow(self, key):
        now = time.time()
        if key not in self.cache or now - self.cache[key] > self.cooldown:
            self.cache[key] = now
            return True
        return False
