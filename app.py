from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from passlib.context import CryptContext
import psycopg2
import os
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime
import ollama
import json

# Настройка пула потоков
executor = ThreadPoolExecutor(max_workers=4)

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'vacmatch'),
    'user': os.getenv('DB_USER', 'vacmatch_user'),
    'password': os.getenv('DB_PASSWORD', 'vacmatchvacmatch'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

pwd_context = CryptContext(schemes=["scrypt"], deprecated="auto")
OLLAMA_MODEL = 'mistral'

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# ========== ИНИЦИАЛИЗАЦИЯ БД ==========
def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(200) NOT NULL
                );
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS vacancy (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(256),
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(64) DEFAULT 'В работе'
                );
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS question (
                    id SERIAL PRIMARY KEY,
                    vacancy_id INTEGER REFERENCES vacancy(id) ON DELETE CASCADE,
                    skill VARCHAR(128),
                    question TEXT,
                    importance INTEGER
                );
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS test_session (
                    id SERIAL PRIMARY KEY,
                    vacancy_id INTEGER REFERENCES vacancy(id),
                    full_name VARCHAR(200),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT
                );
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS answer (
                    id SERIAL PRIMARY KEY,
                    question_id INTEGER REFERENCES question(id),
                    session_id INTEGER REFERENCES test_session(id),
                    answer TEXT,
                    score FLOAT
                );
            ''')
            conn.commit()
    finally:
        conn.close()

init_db()

# ========== ЗАЩИТА ==========
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Войдите в систему", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ========== LLM ==========
def extract_skills_with_llm(text):
    prompt = f"Извлеки ключевые навыки из описания вакансии, навыки напиши на русском:\n{text}\nВерни JSON-массив строк."
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    try:
        return json.loads(response['message']['content'])
    except:
        return ['Python', 'SQL']

def generate_questions_with_llm(skills):
    prompt = f"""
Сгенерируй по одному вопросу на русском языке на каждый навык из списка: {skills}.
Формат JSON: список объектов с полями "skill", "question", "importance" (1–5).
    """.strip()
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    try:
        content = response["message"]["content"]
        data = json.loads(content)
        if isinstance(data, list) and data and "question" in data[0]:
            return data
        raise ValueError("Invalid format")
    except:
        return [{"skill": skill, "question": f"Что вы знаете про {skill}?", "importance": 3} for skill in skills]

def evaluate_answer_with_llm(question, answer):
    # 1. Проверка: если ответ слишком короткий или пустой
    if not answer.strip() or len(answer.strip()) < 10:
        return -1.0

    # 2. Промпт с уточнением
    prompt = f"""
Ты эксперт по найму.
Оцени, насколько ответ кандидата полон, корректен и соответствует вопросу.
- Если ответ пустой, бессмысленный или состоит из одного символа — ставь -1.
- Если ответ частично корректен — от 0 до 0.7 в зависимости от полноты, 0 - это меньше половины верно.
- Если ответ отличный, с примерами — от 0.8 до 1.

Верни **только число** от -1 до 1.

Вопрос: {question}
Ответ: {answer}
"""
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        score_text = response["message"]["content"].strip()
        score = float(score_text.split()[0])
        return min(1.0, max(-1.0, score))  # Ограничение в диапазоне [-1, 1]
    except Exception as e:
        print(f"Ошибка при оценке ответа: {e}")
        return 0.0


def generate_summary_for_candidate(answers, score):
    prompt = f"""Ты HR. Вот ответы кандидата:
{json.dumps([{'question': q, 'answer': a, 'score': s} for q, a, s, _ in answers], ensure_ascii=False)}
Общая оценка: {score}
Сделай краткий отчёт: соответствие, сильные/слабые стороны, рекомендации.
"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# ========== ФОНОВАЯ ГЕНЕРАЦИЯ ==========
def background_generate_questions(vacancy_id, text):
    try:
        skills = extract_skills_with_llm(text)
        questions = generate_questions_with_llm(skills)
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                for q in questions:
                    cur.execute('''
                        INSERT INTO question (vacancy_id, skill, question, importance)
                        VALUES (%s, %s, %s, %s)
                    ''', (vacancy_id, q['skill'], q['question'], q.get('importance', 3)))
                cur.execute("UPDATE vacancy SET status = 'Готово' WHERE id = %s", (vacancy_id,))
    except Exception as e:
        print(f"[Ошибка генерации вопросов для вакансии {vacancy_id}]:", e)
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE vacancy SET status = 'Ошибка' WHERE id = %s", (vacancy_id,))
def background_generate_summary(session_id):
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT q.question, a.answer, a.score, q.importance
                    FROM answer a
                    JOIN question q ON a.question_id = q.id
                    WHERE a.session_id = %s
                """, (session_id,))
                answers = cur.fetchall()
                total_weight = sum([int(row[3]) for row in answers]) or 1
                weighted_score = sum([float(row[2]) * int(row[3]) for row in answers]) / total_weight
                summary = generate_summary_for_candidate(answers, weighted_score)
                cur.execute("UPDATE test_session SET summary = %s WHERE id = %s", (summary, session_id))
    except Exception as e:
        print(f"[Ошибка генерации отчёта по сессии {session_id}]:", e)
# ========== МАРШРУТЫ ==========

@app.route('/')
def home():
    return render_template('home.html', is_authenticated='user_id' in session)
@app.route('/docs')
def docs():
    return render_template('docs.html')
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, title, created_at, status FROM vacancy ORDER BY created_at DESC')
            vacancies = [dict(id=r[0], title=r[1], created_at=r[2], status=r[3]) for r in cur.fetchall()]
        return render_template('dashboard.html', vacancies=vacancies)
    finally:
        conn.close()

@app.route('/vacancy/new', methods=['GET', 'POST'])
@login_required
def new_vacancy():
    if request.method == 'POST':
        text = request.form['text']
        title = text[:60]
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('INSERT INTO vacancy (title, text) VALUES (%s, %s) RETURNING id', (title, text))
                vid = cur.fetchone()[0]
                conn.commit()
                executor.submit(background_generate_questions, vid, text)  # фоновая генерация
                flash('Вакансия создаётся. Вопросы будут сгенерированы позже.', 'info')
                return redirect(url_for('dashboard'))
        finally:
            conn.close()
    return render_template('vacancy_new.html')

@app.route('/test/<int:vacancy_id>', methods=['GET', 'POST'])
def candidate_test(vacancy_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if request.method == 'POST':
                full_name = request.form.get('full_name')
                cur.execute('INSERT INTO test_session (vacancy_id, full_name) VALUES (%s, %s) RETURNING id',
                            (vacancy_id, full_name))
                session_id = cur.fetchone()[0]
                cur.execute('SELECT id, question FROM question WHERE vacancy_id = %s', (vacancy_id,))
                questions = cur.fetchall()
                for qid, text in questions:
                    ans = request.form.get(f'answer{qid}')
                    score = evaluate_answer_with_llm(text, ans)
                    cur.execute('INSERT INTO answer (question_id, session_id, answer, score) VALUES (%s, %s, %s, %s)',
                                (qid, session_id, ans, score))
                conn.commit()
                executor.submit(background_generate_summary, session_id)
                return redirect(url_for('test_result', session_id=session_id))
            else:
                cur.execute('SELECT id, question FROM question WHERE vacancy_id = %s', (vacancy_id,))
                questions = cur.fetchall()
                return render_template('test_candidate_form.html', questions=questions, vacancy_id=vacancy_id)
    finally:
        conn.close()

@app.route('/test/result/<int:session_id>')
def test_result(session_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT q.question, a.answer, a.score, q.importance
                FROM answer a
                JOIN question q ON a.question_id = q.id
                WHERE a.session_id = %s
            """, (session_id,))
            answers = cur.fetchall()

            total_weight = sum([int(row[3]) for row in answers]) or 1
            weighted_score = sum([float(row[2]) * int(row[3]) for row in answers]) / total_weight

            cur.execute("SELECT summary FROM test_session WHERE id = %s", (session_id,))
            summary_row = cur.fetchone()
            summary = summary_row[0] if summary_row and summary_row[0] else "Отчёт ещё формируется..."

            return render_template("test_result.html", answers=answers,
                                   final_score=round(weighted_score, 2),
                                   summary=summary)
    finally:
        conn.close()

@app.route('/vacancy/<int:vacancy_id>/result')
@login_required
def vacancy_result(vacancy_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, full_name, created_at FROM test_session WHERE vacancy_id = %s', (vacancy_id,))
            sessions = cur.fetchall()
            results = []
            for sid, name, dt in sessions:
                cur.execute('''
                    SELECT q.importance, a.score
                    FROM answer a JOIN question q ON a.question_id = q.id
                    WHERE a.session_id = %s
                ''', (sid,))
                data = cur.fetchall()
                total = sum([int(w) for w, _ in data]) or 1
                score = sum([float(s) * int(w) for w, s in data]) / total
                results.append({"name": name, "score": round(score, 2), "date": dt.strftime('%d.%m.%Y'), "session_id": sid})
            return render_template('vacancy_result.html', results=results)
    finally:
        conn.close()

# ========== АВТОРИЗАЦИЯ ==========

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, password FROM users WHERE email = %s', (email,))
                user = cur.fetchone()
                if user and pwd_context.verify(password, user[1]):
                    session['user_id'] = user[0]
                    return redirect(url_for('dashboard'))
                flash("Неверный логин или пароль", "error")
        finally:
            conn.close()
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirmPassword']
        if password != confirm:
            flash("Пароли не совпадают", "error")
            return redirect(url_for('signup'))
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                hashed = pwd_context.hash(password)
                cur.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                            (name, email, hashed))
                conn.commit()
                return redirect(url_for('login'))
        except:
            conn.rollback()
            flash("Ошибка регистрации", "error")
        finally:
            conn.close()
    return render_template('signup.html')
@app.route('/vacancy/<int:vacancy_id>/delete', methods=['POST'])
@login_required
def delete_vacancy(vacancy_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Удалить ответы (answer)
            cur.execute('''
                DELETE FROM answer 
                WHERE question_id IN (
                    SELECT id FROM question WHERE vacancy_id = %s
                )
            ''', (vacancy_id,))

            # Удалить сессии
            cur.execute('DELETE FROM test_session WHERE vacancy_id = %s', (vacancy_id,))

            # Удалить вопросы
            cur.execute('DELETE FROM question WHERE vacancy_id = %s', (vacancy_id,))

            # Удалить вакансию
            cur.execute('DELETE FROM vacancy WHERE id = %s', (vacancy_id,))

            conn.commit()
            flash('Вакансия и все связанные данные удалены.', 'success')
    finally:
        conn.close()

    return redirect(url_for('dashboard'))
@app.route('/vacancy/<int:vacancy_id>/evaluate', methods=['GET', 'POST'])
@login_required
def evaluate_manually(vacancy_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Получаем все вопросы по вакансии
            cur.execute('SELECT id, question FROM question WHERE vacancy_id = %s', (vacancy_id,))
            questions = cur.fetchall()

            if request.method == 'POST':
                cur.execute('INSERT INTO test_session (vacancy_id, full_name) VALUES (%s, %s) RETURNING id',
                            (vacancy_id, 'HR MANUAL'))
                session_id = cur.fetchone()[0]

                for qid, qtext in questions:
                    answer = request.form.get(f'answer{qid}')
                    score = evaluate_answer_with_llm(qtext, answer)
                    cur.execute(
                        'INSERT INTO answer (question_id, session_id, answer, score) VALUES (%s, %s, %s, %s)',
                        (qid, session_id, answer, score)
                    )

                # Генерация отчёта и сохранение
                total_weight = sum([3 for _ in questions]) or 1
                weighted_score = sum([3 * 0.5 for _ in questions]) / total_weight  # для примера
                summary = generate_summary_for_candidate([(q[1], "пример", 0.5, 3) for q in questions], weighted_score)

                cur.execute("UPDATE test_session SET summary = %s WHERE id = %s", (summary, session_id))
                conn.commit()

                return redirect(url_for('test_result', session_id=session_id))

            return render_template('evaluate_vacancy.html', questions=questions, vacancy_id=vacancy_id)
    finally:
        conn.close()

@app.route('/logout')
def logout():
    session.clear()
    flash("Вы вышли из системы", "success")
    return redirect(url_for('home'))

@app.route("/openapi.yaml")
def openapi_spec():
    return send_file("openapi.yaml", mimetype="text/yaml")


# ========== ЗАПУСК ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)