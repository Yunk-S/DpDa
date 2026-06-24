"""
Vercel Python: /api/bnmdap/predict
Serves BNMDAP analysis results from checkpoints.
POST /api/bnmdap/predict
"""
import json
import os

_CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'output/checkpoints')


def handler(event, context):
    hdrs = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
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

        # Parse POST body for risk factor inputs
        try:
            raw = event.get('body', '{}')
            content_type = ''
            hdr = event.get('headers', {}) or {}
            for k, v in hdr.items():
                if k.lower() == 'content-type':
                    content_type = (v or '').lower()
                    break

            if 'application/json' in content_type:
                req_body = json.loads(raw) if raw else {}
            else:
                req_body = {}

            htn = req_body.get('hypertension', 0)
            hd = req_body.get('heart_disease', 0)
            cirr = req_body.get('cirrhosis', 0)
        except Exception:
            htn, hd, cirr = 0, 0, 0

        # Get matching scenario
        scenarios = data.get('scenarios', {})
        scenario_key = f'{htn}_{hd}_{cirr}'
        fallback_key = 'none'

        scenario = scenarios.get(scenario_key, scenarios.get(fallback_key, {}))

        return {'statusCode': 200, 'headers': hdrs,
                'body': json.dumps({
                    'probabilities': scenario,
                    'priors': data.get('priors', {}),
                    'input': {'hypertension': htn, 'heart_disease': hd, 'cirrhosis': cirr}
                })}
    except Exception as e:
        return {'statusCode': 500, 'headers': hdrs,
                'body': json.dumps({'error': str(e)})}
