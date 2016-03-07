import os

from hashlib import sha256


DIRECTORY = os.path.abspath(os.path.dirname(__file__))



SECRET_KEY = os.env.get(
    'SECRET_KEY',
    '\x13\xf4\x95\xb3\x86p\xbf\x1b\xb6B\xc2b'
    '\xf4\x96\xf5\xa78;\x8a+\xf2\xdat\xc2')

SALT = os.env.get('SALT', 'A Default Salt')


