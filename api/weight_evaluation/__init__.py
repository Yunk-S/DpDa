"""
Vercel Python: /api/weight_evaluation/<dataset>
GET /api/weight_evaluation/<dataset> - serve weight evaluation checkpoint
"""
import json
import os

_CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'output/checkpoints')
_ALLOWED = {'stroke', 'heart', 'cirrhosis'}


def handler(event, context):
    hdrs = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': hdrs, 'body': ''}

    path = event.get('path', '')
    parts = [p for p in path.strip('/').split('/') if p]
    # /api/weight_evaluation/stroke -> parts = ['api', 'weight_evaluation', 'stroke']
    dataset = parts[-1] if len(parts) > 2 else None

    if dataset not in _ALLOWED:
        return {
            'statusCode': 400,
            'headers': hdrs,
            'body': json.dumps({'error': f'Unknown dataset: {dataset}'}),
        }

    fpath = os.path.join(_CHECKPOINT_DIR, f'weight_eval_{dataset}.json')
    if not os.path.exists(fpath):
        return {
            'statusCode': 404,
            'headers': hdrs,
            'body': json.dumps({'error': f'No checkpoint for {dataset}'}),
        }

    try:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        return {'statusCode': 200, 'headers': hdrs, 'body': json.dumps(data)}
    except Exception as e:
        return {'statusCode': 500, 'headers': hdrs,
                'body': json.dumps({'error': str(e)})}
