from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def gallery():
    folder = "static/generated/"
    images = sorted(os.listdir(folder))
    images = [f"images/{img}" for img in images if img.endswith(".png")]
    return render_template("index.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)
