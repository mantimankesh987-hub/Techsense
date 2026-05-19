from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pickle
import os
import hashlib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'tech_addict_x9k2p_secret_2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
MODEL_PATH = os.path.join(BASE_DIR, 'knn.pkl')

# Load ML model at startup
import warnings
warnings.filterwarnings('ignore')
model = pickle.load(open(MODEL_PATH, 'rb'))

# ── Database helpers ──────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            result       INTEGER NOT NULL,
            result_label TEXT    NOT NULL,
            age_label    TEXT,
            screen_time  TEXT,
            tech_affect  TEXT,
            improve      TEXT,
            activities   TEXT,
            apps         TEXT,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.close()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'info')
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        action   = request.form.get('action')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('auth'))

        conn = get_db()
        if action == 'register':
            try:
                conn.execute('INSERT INTO users (username,password) VALUES (?,?)',
                             (username, hash_pw(password)))
                conn.commit()
                flash('Account created! Please sign in.', 'success')
            except sqlite3.IntegrityError:
                flash('Username already taken.', 'error')
            finally:
                conn.close()
            return redirect(url_for('auth'))

        elif action == 'login':
            user = conn.execute(
                'SELECT * FROM users WHERE username=? AND password=?',
                (username, hash_pw(password))
            ).fetchone()
            conn.close()
            if user:
                session['user_id']  = user['id']
                session['username'] = user['username']
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'error')
                return redirect(url_for('auth'))

    return render_template('auth.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # ── parse inputs ──
        age          = int(request.form.get('age', 1))          # 0=Under18, 1=18+
        screen_time  = int(request.form.get('screen_time', 1))  # 1-4
        tech_affect  = int(request.form.get('tech_affect', 0))  # 0/1
        improve      = int(request.form.get('improve', 1))      # 0/1

        act_options  = ['Games', 'Social Media', 'Facetime', 'Online Shopping', 'Texting', 'Entertainment']
        app_options  = ['Snapchat', 'Messages', 'Facetime', 'Instagram', 'Facebook',
                        'Tiktok', 'Dating Apps', 'YouTube', 'Netflix', 'WhatsApp',
                        'Reddit', 'Twitter', 'Amazon', 'Discord', 'Spotify']

        activities   = request.form.getlist('activities')
        apps         = request.form.getlist('apps')

        act_values   = [1 if a in activities else 0 for a in act_options]
        app_values   = [1 if a in apps       else 0 for a in app_options]

        # Feature vector: [Age, ScreenTime, TechAffect, Improve, Act1-6, App1-15] = 25
        features = [age, screen_time, tech_affect, improve] + act_values + app_values

        prediction    = int(model.predict([features])[0])
        result_label  = "Technologically Addicted" if prediction == 1 else "Not Addicted"

        age_label     = 'Under 18'  if age == 0           else '18 and above'
        st_label      = ['0-2 hrs', '2-5 hrs', '5-7 hrs', '7+ hrs'][screen_time - 1]
        ta_label      = 'Yes' if tech_affect == 1 else 'No'
        imp_label     = 'Yes' if improve      == 1 else 'No'

        conn = get_db()
        conn.execute('''
            INSERT INTO predictions
              (user_id,result,result_label,age_label,screen_time,tech_affect,improve,activities,apps)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', (
            session['user_id'], prediction, result_label,
            age_label, st_label, ta_label, imp_label,
            ', '.join(activities) if activities else 'None',
            ', '.join(apps)       if apps       else 'None'
        ))
        conn.commit()
        conn.close()

        return render_template('result.html',
            prediction   = prediction,
            result_label = result_label,
            age_label    = age_label,
            st_label     = st_label,
            ta_label     = ta_label,
            imp_label    = imp_label,
            activities   = ', '.join(activities) if activities else 'None',
            apps         = ', '.join(apps)       if apps       else 'None'
        )

    return render_template('predict.html')


@app.route('/dashboard')
@login_required
def dashboard():
    conn    = get_db()
    rows    = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()

    total       = len(rows)
    addicted    = sum(1 for r in rows if r['result'] == 1)
    not_addicted = total - addicted

    return render_template('dashboard.html',
        predictions  = rows,
        total        = total,
        addicted     = addicted,
        not_addicted = not_addicted
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
