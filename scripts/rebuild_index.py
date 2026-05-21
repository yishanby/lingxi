import asyncio, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from app.services.rag import build_index

async def main():
    print('Rebuilding RAG index with bge-small-zh-v1.5...', flush=True)
    index = await build_index(9)
    n_msgs = index['indexed_messages']
    n_chunks = len(index['chunks'])
    print(f'Done: {n_msgs} msgs, {n_chunks} chunks', flush=True)

asyncio.run(main())
