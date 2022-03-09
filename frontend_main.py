from flask import Flask, render_template, request, jsonify
import sys
from flask_jsglue import JSGlue
# from main import test

app = Flask(__name__)
jsglue = JSGlue(app)

@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html', content='John')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/preparation')
def preparation():
    return render_template('preparation.html')

@app.route('/pc1')
def pc1():
    return render_template('pc1/index.html')

@app.route('/rundetection') 
def run_detection(): 
    print("Running test...")
    # test()

@app.route("/function_route", methods=["GET", "POST"])
def my_function():
    if request.method == "POST":
        print("POST request")
        data = {}    # empty dict to store data
        # data['title'] = request.json['title']
        # data['release_date'] = request.json['movie_release_date']

       # do whatever you want with the data here e.g look up in database or something
       # if you want to print to console

        print(request.json, file=sys.stderr)

        # then return something back to frontend on success

        # this returns back received data and you should see it in browser console
        # because of the console.log() in the script.
        # return jsonify(request)
        return jsonify("Success")
    else:
        return render_template('the_page_i_was_on.html')

if __name__ == '__main__':
    app.run()