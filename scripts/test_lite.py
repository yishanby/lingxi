import httpx, time
t = time.time()
r = httpx.get('http://127.0.0.1:8000/api/sessions?lite=true', timeout=60)
print(f'status={r.status_code} took {time.time()-t:.1f}s size={len(r.content)}B')
print(r.text[:500])
