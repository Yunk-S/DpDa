"""
Vercel Python: /api/bnmdap/predict
Serves BNMDAP analysis results from checkpoints.
GET /api/bnmdap/predict
"""
import json
import os

_CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'output/checkpoints')

def handler(event, context):
    hdrs = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': hdrs, 'body': ''}

    fpath = os.path.join(_CHECKPOINT_DIR, 'bnmdap_analysis.json')
    if not os.path.exists(fpath):
        return {'statusCode': 404, 'headers': hdrs,
                'body': json.dumps({'error': 'BNMDAP checkpoint not found'})}

    try:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        return {'statusCode': 200, 'headers': hdrs,
                'body': json.dumps(data)}
    except Exception as e:
        return {'statusCode': 500, 'headers': hdrs,
                'body': json.dumps({'error': str(e)})}
