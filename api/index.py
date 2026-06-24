"""
Vercel serverless index - serves the Flask app as a serverless function.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Vercel serverless environment may not have Flask templates
# Instead, redirect all non-API routes to a simple health/info response
def handler(event, context):
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}

    import json

    path = event.get('path', '/')

    if path in ('/', '/index', '/health'):
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'status': 'ok',
                'service': 'DpDa Disease Prediction API',
                'version': '1.0',
                'endpoints': {
                    'POST /api/predict': 'Disease prediction (stroke | heart | cirrhosis)',
                    'GET /api/health': 'Health check',
                }
            }),
        }

    return {
        'statusCode': 404,
        'headers': headers,
        'body': json.dumps({'error': 'Not found. Use POST /api/predict'}),
    }
