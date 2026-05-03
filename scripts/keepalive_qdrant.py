import os
from qdrant_client import QdrantClient

host = os.environ['QDRANT_HOST']

# No API key — connect without auth
client = QdrantClient(url=f"https://{host}")

collections = client.get_collections()
print(f"✓ Qdrant keep-alive ping — {len(collections.collections)} collections active")
