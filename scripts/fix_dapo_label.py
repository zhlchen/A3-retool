import json

src = '/home/c00954327/slime-run/data/dapo-math-17k.jsonl'
dst = '/home/c00954327/slime-run/data/dapo-math-17k-label.jsonl'

n = 0
with open(src) as fi, open(dst, 'w') as fo:
    for line in fi:
        d = json.loads(line)
        d['label'] = d['reward_model']['ground_truth']
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        n += 1
print('records:', n)
