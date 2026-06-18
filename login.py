import base64
import os
import joblib
import pandas as pd
import sqlite3
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Career Prediction System", page_icon="🎓", layout="centered"
)


# =====================================
# FUNCTION TO CONVERT IMAGE TO BASE64
# =====================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""


IMAGE_FILENAME = "ila.jpg"
img_base64 = get_base64_image(IMAGE_FILENAME)

if img_base64:
    background_style = f"""
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)), url("data:image/jpeg;base64,{img_base64}");
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    """
else:
    background_style = """
    .stApp {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
        font-family: 'Segoe UI', sans-serif;
    }
    """

# =====================================
# CUSTOM CSS 
# =====================================
st.markdown(
    f"""
<style>
{background_style}

/* ================= MAIN FORM CARD ================= */
[data-testid="stForm"] {{
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    border: 3px solid #2563eb !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4) !important;
}}

/* ================= 🎯 BROWSE FILES BUTTON BLUE COLOUR FIXED 🎯 ================= */
div[data-testid="stFileUploader"] {{
    background: rgba(255, 255, 255, 0.95) !important;
    padding: 20px !important;
    border-radius: 15px !important;
    border: 2px dashed #2563eb !important;
}}

div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {{
    background-color: #2563eb !important;
    color: white !important;
    border: 2px solid #1e40af !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    text-transform: none !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 5px 15px rgba(37,99,235,0.3) !important;
}}

div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {{
    background-color: #1e40af !important;
    border-color: #1e3a8a !important;
    color: white !important;
    box-shadow: 0 6px 20px rgba(30,64,175,0.5) !important;
    transform: translateY(-1px);
}}

div[data-testid="stFileUploader"] section div,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploaderDropzone"] p {{
    color: #1e3a8a !important;
    font-weight: 600 !important;
}}

/* ================= LABEL STYLE ================= */
label,
[data-testid="stWidgetLabel"] {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

label p,
[data-testid="stWidgetLabel"] p {{
    color: #1e3a8a !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}}

/* ================= 🎯 USERNAME & PASSWORD LABELS ONLY IN WHITE 🎯 ================= */
div[data-testid="stTextInput"] label p {{
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8) !important;
}}

/* ================= INPUT BOXES ================= */
div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div,
div[data-baseweb="select"] > div {{
    border: 2px solid #1e40af !important;
    border-radius: 10px !important;
    background: white !important;
}}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
    color: #334155 !important;
    background: white !important;
    padding: 12px !important;
}}

div[data-testid="stNumberInput"] button {{
    display: none !important;
}}

/* ================= ALL OTHER BUTTONS ================= */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {{
    background: #2563eb !important;
    color: white !important;
    border: 2px solid #1e40af !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    height: 50px !important;
    width: 100% !important;
    box-shadow: 0 6px 15px rgba(37,99,235,0.3) !important;
}}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {{
    background: #1e40af !important;
    border-color: #1e3a8a !important;
    color: white !important;
}}

/* ================= TITLE & TEXT REVERB ================= */
h1 {{
    color: #ffffff !important;
    text-align: center !important;
    font-weight: 800 !important;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.8) !important;
}}
h2, h3, h4, h5, p, span {{
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
}}

div[data-testid="stDataFrame"] {{
    background: rgba(255, 255, 255, 0.9) !important;
    padding: 10px !important;
    border-radius: 10px !important;
}}

/* ================= ALERT CONTAINER RULES ================= */
div[data-testid="stAlertContainer"] > div[aria-label="Success"] {{
    background-color: #10b981 !important;
    border-radius: 8px !important;
}}
div[data-testid="stAlertContainer"] > div[aria-label="Success"] p {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

div[data-testid="stAlertContainer"] > div[aria-label="Error"] {{
    background-color: #ef4444 !important;
    border-radius: 8px !important;
}}
div[data-testid="stAlertContainer"] > div[aria-label="Error"] p {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

[data-testid="stInputInstruction"] {{
    display: none !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.components.v1.html(
    """
    <script>
        parent.document.querySelectorAll('input').forEach(el => {
            el.setAttribute('autocomplete', 'off');
        });
    </script>
    """,
    height=0,
)

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
"""
)
conn.commit()

# =========================
# MODEL
# =========================
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "username" not in st.session_state:
    st.session_state.username = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "encoders" not in st.session_state:
    st.session_state.encoders = {}
if "ready" not in st.session_state:
    st.session_state.ready = False


# =========================
# DB FUNCTIONS
# =========================
def register_user(username, password):
    try:
        c.execute(
            "INSERT INTO users(username, password) VALUES (?,?)",
            (username, password),
        )
        conn.commit()
        return True
    except:
        return False


def login_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password),
    )
    return c.fetchone()


# =========================
# SIDEBAR
# =========================
option = st.sidebar.selectbox("Choose Option", ["Login", "Register"])


# =========================
# LOGIN PAGE
# =========================
def login_page():
    st.title("🔐 Career Prediction System")

    if option == "Login":
        st.subheader("Login Form")
        username = st.text_input(
            "Username", key="login_user", label_visibility="visible"
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_pass",
            label_visibility="visible",
        )

        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.page = "upload"
                st.session_state.username = username
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

    elif option == "Register":
        st.subheader("Registration Form")
        new_user = st.text_input(
            "New Username", key="reg_user", label_visibility="visible"
        )
        new_pass = st.text_input(
            "New Password",
            type="password",
            key="reg_pass",
            label_visibility="visible",
        )

        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Account Created Successfully ✅")
            else:
                st.error("User already exists")


# =====================================
# UPLOAD + AUTO PIPELINE
# =====================================
def upload_page():
    st.title("📁 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("File Uploaded Successfully")

        # ==============================
        # 📊 ANALYSIS
        # ==============================
        st.subheader("📊 Shape")
        st.write(df.shape)

        st.subheader("📌 Head")
        st.dataframe(df.head())

        st.subheader("❗ Missing Values")
        st.dataframe(df.isnull().sum())

        st.subheader("♻️ Duplicates")
        st.write(df.duplicated().sum())

        # ==============================
        # 🧹 CLEAN DATA
        # ==============================
        df = df.dropna()
        df = df.drop_duplicates()

        st.session_state.df = df

        st.subheader("🧹 Clean Data")
        st.dataframe(df)

        # ============================================================
        # 🤖 TRAIN MODEL 
        # ============================================================
        try:
            target = "Future Career"

            df_encoded = df.copy()
            encoders = {}

            for col in df_encoded.columns:
                if df_encoded[col].dtype == "object":
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    encoders[col] = le

            st.session_state.encoders = encoders

            X = df_encoded.drop(columns=[target])
            y = df_encoded[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            model_name = type(model).__name__

            st.subheader("🤖 Model Training Results")

            # Green Banner for Model Results
            st.markdown(
                f"""
           <div style="
               background: #10b981 !important; 
               background-color: #10b981 !important; 
               padding: 20px; 
               border-radius: 8px; 
               box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
               margin-top: 15px;
               margin-bottom: 25px;
               width: 100%;
           ">
               <p style="color: #ffffff !important; font-size: 17px; font-weight: 700; margin: 8px 0; font-family: 'Segoe UI', sans-serif;">
                   📦 Model Name: <span style="color: #ffffff !important; font-weight: 600;">{model_name}</span>
               </p>
               <p style="color: #ffffff !important; font-size: 17px; font-weight: 700; margin: 8px 0; font-family: 'Segoe UI', sans-serif;">
                   🎯 Model Accuracy: <span style="color: #ffffff !important; font-weight: 600;">{acc:.4f}</span>
               </p>
           </div>
           """,
                unsafe_allow_html=True,
            )

            st.session_state.ready = True

        except Exception as e:
            st.error(f"Training Error: {e}")

    if st.session_state.ready:
        if st.button("🚀 Go to Predict Page"):
            st.session_state.page = "predict"
            st.rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.username = ""
        st.rerun()


# =====================================
# PREDICT PAGE 
# =====================================
def predict_page():
    st.title("🎯 Career Prediction System")
    username = st.session_state.get("username", "User")
    st.success(f"Welcome, {username}!")

    df = st.session_state.df
    encoders = st.session_state.encoders

    if df is None:
        st.warning("Upload dataset first")
        return

    if st.button("⬅️ Back"):
        st.session_state.page = "upload"
        st.rerun()

    with st.form("career_form"):
        st.subheader("👨‍🎓 Student Information")
        col1, col2 = st.columns(2)

        with col1:
            student_id = st.number_input("Student ID", min_value=1, value=1)
            age = st.number_input("Age", min_value=15, value=18)
            gender = st.selectbox("Gender", encoders["Gender"].classes_)
            gpa = st.number_input(
                "GPA", min_value=0.0, max_value=10.0, value=5.0
            )

        with col2:
            major = st.selectbox("Major", encoders["Major"].classes_)
            domain = st.selectbox(
                "Interested Domain", encoders["Interested Domain"].classes_
            )
            projects = st.selectbox("Projects", encoders["Projects"].classes_)

        st.subheader("💻 Programming Skills")
        col3, col4, col5 = st.columns(3)

        with col3:
            python_skill = st.selectbox("Python", encoders["Python"].classes_)
        with col4:
            sql_skill = st.selectbox("SQL", encoders["SQL"].classes_)
        with col5:
            java_skill = st.selectbox("Java", encoders["Java"].classes_)

        submit = st.form_submit_button("🚀 Predict Career")

    if submit:
        try:
            input_data = pd.DataFrame(
                [
                    {
                        "Student ID": student_id,
                        "Gender": encoders["Gender"].transform([gender])[0],
                        "Age": age,
                        "GPA": gpa,
                        "Major": encoders["Major"].transform([major])[0],
                        "Interested Domain": encoders["Interested Domain"].transform(
                            [domain]
                        )[
                            0
                        ],
                        "Projects": encoders["Projects"].transform([projects])[
                            0
                        ],
                        "Python": encoders["Python"].transform([python_skill])[
                            0
                        ],
                        "SQL": encoders["SQL"].transform([sql_skill])[0],
                        "Java": encoders["Java"].transform([java_skill])[0],
                    }
                ]
            )

            pred = model.predict(input_data)[0]
            career = encoders["Future Career"].inverse_transform([pred])[0]
            st.success(f"🎯 Predicted Career: {career}")
        except Exception as e:
            st.error(f"Prediction Mapping Error: {e}")


# =====================================
# ROUTING
# =====================================
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "predict":
        predict_page()