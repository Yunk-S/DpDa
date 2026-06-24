"""
Pre-render Flask templates to static HTML for Vercel deployment.
Outputs rendered HTML files to repo root.
Vercel serves them as static files at /, /predict, /weight-evaluation, etc.
"""
import os
import sys
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from flask import Flask, render_template

app = Flask(__name__, template_folder=os.path.join(ROOT, 'templates'))
app.config['TEMPLATES_AUTO_RELOAD'] = False


@app.route('/')
def index(): return render_template('index.html')


@app.route('/predict')
@app.route('/predict/<disease>')
def predict(disease=None): return render_template('predict.html', disease=disease or 'stroke')


@app.route('/weight-evaluation')
@app.route('/weight-evaluation/<ds>')
def weight_evaluation(ds=None): return render_template('weight_evaluation.html', dataset=ds or 'heart')


@app.route('/awelm')
@app.route('/awelm/<ds>')
def awelm_page(ds=None): return render_template('awelm.html', dataset=ds or 'heart')


@app.route('/bnmdap')
def bnmdap(): return render_template('bnmdap.html')


@app.route('/multi-predict')
def multi_predict(): return render_template('multi_predict.html')


@app.route('/multi-disease')
def multi_disease(): return render_template('multi_disease.html')


@app.route('/data-analysis')
def data_analysis(): return render_template('data_analysis.html')


@app.route('/model-performance')
def model_performance(): return render_template('model_performance.html')


if __name__ == '__main__':
    pages = [
        ('/', 'index.html'),
        ('/predict', 'predict.html'),
        ('/weight-evaluation', 'weight_evaluation.html'),
        ('/awelm', 'awelm.html'),
        ('/bnmdap', 'bnmdap.html'),
        ('/multi-predict', 'multi_predict.html'),
        ('/multi-disease', 'multi_disease.html'),
        ('/data-analysis', 'data_analysis.html'),
        ('/model-performance', 'model_performance.html'),
    ]

    # Clean up previous public/ directory
    pub = os.path.join(ROOT, 'public')
    if os.path.exists(pub):
        shutil.rmtree(pub)

    with app.test_client() as client:
        for url, filename in pages:
            print(f'Rendering {url} -> {filename} ...', end=' ')
            resp = client.get(url)
            if resp.status_code == 200:
                out_path = os.path.join(ROOT, filename)
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(resp.get_data(as_text=True))
                print(f'OK ({len(resp.data):,} bytes)')
            else:
                print(f'ERROR {resp.status_code}')

    print('\nStatic HTML pages written to repo root.')
    print('Vercel will serve them at /, /predict, etc. (no .html extension needed)')
