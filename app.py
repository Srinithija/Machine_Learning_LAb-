from flask import Flask, render_template, request
from pca_model import perform_pca

app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template('index.html', selected_components=2)

# Run PCA
@app.route('/run_pca', methods=['POST'])
def run_pca():

    n_components = int(request.form['components'])

    # Perform PCA
    pca_df, variance = perform_pca(n_components)

    # Convert variance to percentage
    variance_percent = [round(v * 100, 2) for v in variance]

    return render_template(
        'index.html',
        variance=variance_percent,
        data=pca_df.head().to_html(classes='data', index=False),
        selected_components=n_components
    )

if __name__ == '__main__':
    app.run(debug=True)