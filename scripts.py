from flask import Flask
from scripts import model
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
model.load_weights("model_transformer.h5")

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(host='0.0.0.0', port=80)