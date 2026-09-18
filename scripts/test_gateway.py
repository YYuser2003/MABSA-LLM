import sys, os
sys.path.insert(0, os.path.abspath('.'))
from bacr.client import GeminiClient

client = GeminiClient(allow_fallback=False)
res = client.call_text('You are a test assistant.', 'Output strictly valid JSON: {"status": "ok"}')
print('Gateway response:', res)
