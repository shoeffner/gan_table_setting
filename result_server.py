from flask import Flask, render_template, send_from_directory


app = Flask(__name__)


@app.route('/')
def index():
    with open('results/runs.list') as f:
        runs = reversed(list(l for l in f.read().strip().splitlines() if l))
    return render_template('index.html', runs=runs)


@app.route('/run/<run>')
def run(run):
    return send_from_directory(f'results/{run}', 'index.html')


@app.route('/results/<path:path>')
def files(path):
    return send_from_directory('results', path)


if __name__ == '__main__':
    app.run(debug=True)
