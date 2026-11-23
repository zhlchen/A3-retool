import json

src = '/home/c00954327/slime-run/data/dapo-math-17k-label.jsonl'
dst = '/home/c00954327/slime-run/data/dapo-retool.jsonl'

n = 0
with open(src) as fi, open(dst, 'w') as fo:
    for line in fi:
        d = json.loads(line)
        # retool 模板要纯字符串 prompt(官方 rl_data_preprocess.py 同款拍平)
        prompt = d['prompt'][0]['content'] if isinstance(d['prompt'], list) else d['prompt']
        fo.write(json.dumps({'prompt': prompt, 'label': d['label']}, ensure_ascii=False) + '\n')
        n += 1
print('records:', n)
