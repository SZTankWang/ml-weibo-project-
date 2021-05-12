from flask import Flask,request
from flask import render_template
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def landing():
	return render_template("index.html")

@app.route("/newBlog")
def receiveBlog():
	text = request.args.get("text")
	print(text)
	res = {"code":200,"likes":10000,"forwards":1000,"comments":2000}
	return jsonify(res)

