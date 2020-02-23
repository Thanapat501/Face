from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

from flask import Flask

app = Flask(__name__)


@app.route('/example/')
def example():
    return {'hello': 'world'}


if __name__ == "__main__":
    app.run(debug=True)