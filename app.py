import os
import logging
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime

from helpers import apology, login_required, lookup, inr, predict_price

# ------------------- CONFIGURATION -------------------
app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Auto-reload templates
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Session
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Database
db = SQL("sqlite:///finance.db")

# Ensure API key is set
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not api_key:
    raise RuntimeError("ALPHA_VANTAGE_API_KEY not set in environment")

# ------------------- ROUTES -------------------

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        session.clear()

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Provide username and password.")
            return redirect("/login")

        rows = db.execute("SELECT * FROM users WHERE username = ?", username)
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            flash("Invalid username/password.")
            return redirect("/login")

        session["user_id"] = rows[0]["id"]
        return redirect("/")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/home")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username or not password or not confirmation:
            flash("All fields are required.")
            return redirect("/register")
        if password != confirmation:
            flash("Passwords do not match.")
            return redirect("/register")
        if db.execute("SELECT * FROM users WHERE username = ?", username):
            flash("Username already taken.")
            return redirect("/register")

        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, generate_password_hash(password))
        flash("Registration successful! Login now.")
        return redirect("/login")
    return render_template("register.html")

@app.route("/")
@login_required
def index():
    rows = db.execute("SELECT * FROM stocks WHERE user_id = ?", session["user_id"])
    portfolios = []
    stock_value = 0

    for row in rows:
        stock_info = lookup(row["symbol"])
        if stock_info:
            name = stock_info["name"]
            amount = row["shares"]
            current = stock_info["price"]
            value = round(current * amount, 2)
            portfolios.append([row["symbol"], name, amount, inr(current), inr(value)])
            stock_value += value

    cash_balance = round(db.execute("SELECT cash FROM users WHERE id = ?", session["user_id"])[0]["cash"], 2)
    stock_value = round(stock_value, 2)
    grand_total = round(cash_balance + stock_value, 2)
    start_value = 10000
    profit_loss = round(grand_total - start_value, 2)

    return render_template("portfolio.html", portfolios=portfolios, cash_balance=inr(cash_balance),
                           stock_value=inr(stock_value), grand_total=inr(grand_total),
                           start_value=inr(start_value), end_value=inr(grand_total), profit_loss=inr(profit_loss))

# --------- Other routes like /buy, /sell, /quote, /history, /predict ---------
# Keep your existing implementations as is; they are compatible.

# ------------------- ERROR HANDLER -------------------
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError

def errorhandler(e):
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)

for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

# ------------------- RUN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
