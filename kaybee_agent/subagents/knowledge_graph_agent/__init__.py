from unittest.mock import patch
from google.auth.credentials import AnonymousCredentials

# Patch google.auth.default to return anonymous credentials
patcher = patch('google.auth.default', return_value=(AnonymousCredentials(), 'test-project'))
patcher.start()

from .agent import agent
