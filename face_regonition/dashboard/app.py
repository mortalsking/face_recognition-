from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def home():
    faces = os.listdir("faces")
    return render_template("index.html", faces=faces)

if __name__ == "__main__":
    app.run(debug=True)

