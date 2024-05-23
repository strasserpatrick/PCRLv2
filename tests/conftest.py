import sys


def random_fn(lower, _):
    return lower


module = type(sys)('numpy')
module.random = type('random', (), {'randint': random_fn})
sys.modules['numpy'] = module
