import streamlit as st
import pandas as pd
# import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_login_email(user_email):
    try:
        # Check if email configuration is set
        if not EMAIL_USERNAME or not EMAIL_PASSWORD:
            st.error("Email configuration is missing. Please check your .env file.")
            return False

        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = user_email
        msg['Subject'] = "Successful Login - Heart Disease Prediction System"

        # Email body
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #ff4b4b;">Login Successful!</h2>
                    <p>Dear User,</p>
                    <p>You have successfully logged into the Heart Disease Prediction System.</p>
                    <p style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                        If this was not you, please contact our support team immediately at {EMAIL_USERNAME}.
                    </p>
                    <br>
                    <p>Best regards,</p>
                    <p>Heart Disease Prediction System Team</p>
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        This is an automated message. Please do not reply to this email.
                    </p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        try:
            # Create SMTP session
            st.info("Connecting to SMTP server...")
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            
            # Login to SMTP server
            st.info("Authenticating with SMTP server...")
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            
            # Send email
            st.info("Sending email...")
            server.send_message(msg)
            server.quit()
            
            st.success("Email sent successfully!")
            return True
            
        except smtplib.SMTPAuthenticationError:
            st.error("Failed to authenticate with Gmail. Please check your email and app password in the .env file.")
            return False
        except smtplib.SMTPException as smtp_error:
            st.error(f"SMTP error occurred: {str(smtp_error)}")
            return False
            
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        st.info("Please make sure you have:")
        st.info("1. Created a .env file with your Gmail credentials")
        st.info("2. Generated an App Password from your Google Account")
        st.info("3. Enabled 'Less secure app access' in your Google Account settings")
        return False

# Set page config
st.set_page_config(
    page_title="Cardiovascular Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main title styling */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Card styling */
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* Radio button styling */
    .stRadio>div>div {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #1f1f1f;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 10px;
    }
    
    /* Plot styling */
    .plot-container {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Database Setup --------------------
def create_tables():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Create users table if not exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT,
                        email TEXT UNIQUE)''')

    # Create feedback table if not exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        feedback TEXT)''')
    
    conn.commit()
    conn.close()

conn = sqlite3.connect('users.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                email TEXT UNIQUE)''')

conn.commit()

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()

# -------------------- Train Model --------------------
@st.cache_data
def train_model():
    X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

# -------------------- Prediction --------------------
def predict_heart_disease(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# -------------------- Navbar --------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu", 
        options=["Home", "Predict", "EDA", "Medical History", "Feedback"], 
        icons=["house", "activity", "bar-chart", "book", "chat-right-text"],
        menu_icon="cast", 
        default_index=0
    )

# -------------------- User Authentication --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        result = c.fetchone()
        
        if result:
            st.session_state.logged_in = True
            st.session_state.username = username
            
            # Get user's email from database
            c.execute("SELECT email FROM users WHERE username = ?", (username,))
            user_email = c.fetchone()[0]
            
            # Send login notification email
            if send_login_email(user_email):
                st.success("Login successful! A confirmation email has been sent to your registered email address.")
            else:
                st.success("Login successful! (Email notification failed)")
        else:
            st.error("Invalid username or password")

def register():
    st.subheader("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_email = st.text_input("Email Address")
    
    if st.button("Register"):
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                     (new_username, new_password, new_email))
            conn.commit()
            st.success("Registration successful! Please login.")
        except sqlite3.IntegrityError:
            st.error("Username or email already exists. Please try a different one.")

if not st.session_state.logged_in:
    st.title("Welcome to Cardiovascular Disease Prediction System")
    
    # Create a container for login/register
    auth_container = st.container()
    
    with auth_container:
        # Create tabs for login and register
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login()
        
        with tab2:
            register()
    
    st.stop()

# -------------------- Home Section --------------------
if selected == "Home":
    # Add a custom header with gradient background
    st.markdown("""
        <div style='background: linear-gradient(to right, #ff4b4b, #ff6b6b); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0;'>🏥 Cardiovascular Disease Prediction System</h1>
        </div>
    """, unsafe_allow_html=True)

    # Add an image from an online source with custom styling
    st.markdown("""
        <div style='border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
    """, unsafe_allow_html=True)
    st.image("https://source.unsplash.com/800x400/?heart,health", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Welcome message with custom styling
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
            <h3 style='color: #1f1f1f; margin-bottom: 1rem;'>💖 Welcome to the Cardiovascular Disease Prediction System!</h3>
            <p style='color: #666;'>This application helps you predict the likelihood of heart disease based on several health parameters.</p>
            <div style='margin-top: 1rem;'>
                <h4 style='color: #1f1f1f;'>Key Features:</h4>
                <ul style='list-style-type: none; padding: 0;'>
                    <li style='margin: 0.5rem 0;'>🌡️ Real-time prediction using Machine Learning</li>
                    <li style='margin: 0.5rem 0;'>📊 Data visualization and analysis</li>
                    <li style='margin: 0.5rem 0;'>🏥 Doctor search and booking</li>
                    <li style='margin: 0.5rem 0;'>💬 User feedback system</li>
                    <li style='margin: 0.5rem 0;'>📱 Interactive health monitoring</li>
                    <li style='margin: 0.5rem 0;'>🎯 Personalized health recommendations</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Health Statistics with custom styling
    st.markdown("""
        <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #1f1f1f; margin-bottom: 1.5rem;'>📊 Global Heart Disease Statistics</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Deaths", "17.9M", "↑ 2.1%", delta_color="inverse")
    with col2:
        st.metric("Risk Factor Prevalence", "85%", "↑ 1.5%", delta_color="inverse")
    with col3:
        st.metric("Preventable Cases", "80%", "↓ 0.5%", delta_color="normal")
    st.markdown("</div>", unsafe_allow_html=True)

    # Health Tips with custom styling
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
            <h3 style='color: #1f1f1f; margin-bottom: 1.5rem;'>💡 Daily Health Tips</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;'>
    """, unsafe_allow_html=True)
    
    tips = [
        "🏃‍♂️ Exercise for at least 30 minutes daily",
        "🥗 Eat a balanced diet rich in fruits and vegetables",
        "💧 Stay hydrated - drink 8 glasses of water daily",
        "😴 Get 7-8 hours of quality sleep",
        "🧘‍♂️ Practice stress management techniques",
        "🚭 Avoid smoking and limit alcohol consumption",
        "🩺 Regular health check-ups are essential",
        "🧠 Stay mentally active and socially connected"
    ]
    
    for tip in tips:
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 1rem; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                {tip}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Risk Factors with custom styling
    st.markdown("""
        <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #1f1f1f; margin-bottom: 1.5rem;'>⚠️ Common Risk Factors</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;'>
    """, unsafe_allow_html=True)
    
    risk_factors = {
        "High Blood Pressure": "Affects 1 in 3 adults",
        "High Cholesterol": "Leading cause of heart disease",
        "Diabetes": "Doubles heart disease risk",
        "Obesity": "Increases risk by 40%",
        "Smoking": "Major preventable cause",
        "Physical Inactivity": "Affects 1 in 4 adults"
    }
    
    for factor, stat in risk_factors.items():
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px;'>
                <strong style='color: #1f1f1f;'>{factor}:</strong><br>
                <span style='color: #666;'>{stat}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Emergency Information with custom styling
    st.markdown("""
        <div style='background-color: #fff3cd; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
            <h3 style='color: #856404; margin-bottom: 1.5rem;'>🚨 Emergency Warning Signs</h3>
            <div style='color: #856404;'>
                <p style='margin-bottom: 1rem;'>Seek immediate medical attention if you experience:</p>
                <ul style='list-style-type: none; padding: 0;'>
                    <li style='margin: 0.5rem 0;'>• Severe chest pain or pressure</li>
                    <li style='margin: 0.5rem 0;'>• Shortness of breath</li>
                    <li style='margin: 0.5rem 0;'>• Pain in arms, back, neck, or jaw</li>
                    <li style='margin: 0.5rem 0;'>• Cold sweats</li>
                    <li style='margin: 0.5rem 0;'>• Nausea or lightheadedness</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Disclaimer with custom styling
    st.markdown("""
        <div style='background-color: #cce5ff; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
            <h4 style='color: #004085; margin-bottom: 1rem;'>⚠️ Important Disclaimer</h4>
            <p style='color: #004085; margin: 0;'>
                This app is for informational purposes only. Always consult with healthcare professionals for medical advice and diagnosis.
                In case of emergency, call your local emergency services immediately.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------- Prediction Section --------------------
if selected == "Predict":
    st.title("🩺 Predict Cardiovascular Disease")
    
    # Add BMI Calculator
    st.subheader("📊 BMI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", 30, 200, 70)
    with col2:
        height = st.number_input("Height (cm)", 100, 250, 170)
    
    if st.button("Calculate BMI"):
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        st.metric("Your BMI", f"{bmi:.1f}")
        if bmi < 18.5:
            st.warning("Underweight - Consider consulting a nutritionist")
        elif bmi < 25:
            st.success("Normal weight - Keep maintaining a healthy lifestyle!")
        elif bmi < 30:
            st.warning("Overweight - Consider increasing physical activity")
        else:
            st.error("Obese - Please consult a healthcare provider")

    # Add information about chest pain types
    st.subheader("Understanding Chest Pain Types")
    chest_pain_info = {
        0: "Typical Angina: Chest pain or discomfort that occurs when the heart muscle doesn't get enough oxygen-rich blood. Usually feels like pressure, squeezing, or fullness in the chest.",
        1: "Atypical Angina: Similar to typical angina but with different characteristics. Pain may be less severe or occur in different locations.",
        2: "Non-Anginal Pain: Chest pain that is not related to heart problems. Often caused by muscle strain, acid reflux, or other conditions.",
        3: "Asymptomatic: No chest pain or discomfort, but other symptoms may be present."
    }
    
    st.markdown("### Chest Pain Types and Symptoms")
    for pain_type, description in chest_pain_info.items():
        with st.expander(f"Chest Pain Type {pain_type}"):
            st.write(description)
            if pain_type == 0:  # Typical Angina
                st.markdown("**Common Symptoms:**")
                st.markdown("- Pressure or tightness in the chest")
                st.markdown("- Pain that may spread to the arms, neck, jaw, or back")
                st.markdown("- Shortness of breath")
                st.markdown("- Nausea or dizziness")
            elif pain_type == 1:  # Atypical Angina
                st.markdown("**Common Symptoms:**")
                st.markdown("- Pain in the upper abdomen")
                st.markdown("- Pain in the back, neck, or jaw")
                st.markdown("- Fatigue or weakness")
                st.markdown("- Sweating")
            elif pain_type == 2:  # Non-Anginal Pain
                st.markdown("**Common Symptoms:**")
                st.markdown("- Sharp or stabbing pain")
                st.markdown("- Pain that worsens with movement")
                st.markdown("- Pain that changes with breathing")
                st.markdown("- Burning sensation")
            else:  # Asymptomatic
                st.markdown("**Warning Signs to Watch For:**")
                st.markdown("- Shortness of breath")
                st.markdown("- Fatigue")
                st.markdown("- Irregular heartbeat")
                st.markdown("- Dizziness or lightheadedness")

    st.markdown("---")
    st.subheader("Enter Your Health Information")

    # Add tooltips and help text for each input
    user_input = {
        'age': st.number_input("Age", 20, 100, 50, help="Enter your current age"),
        'sex': st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", help="Select your biological sex"),
        'cp': st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x], help="Select the type of chest pain you experience"),
        'trestbps': st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120, help="Enter your resting blood pressure"),
        'chol': st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, help="Enter your cholesterol level"),
        'fbs': st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Is your fasting blood sugar above 120 mg/dl?"),
        'restecg': st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x], help="Select your ECG result"),
        'thalach': st.number_input("Maximum Heart Rate Achieved (bpm)", 60, 220, 150, help="Enter your maximum heart rate during exercise"),
        'exang': st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you experience chest pain during exercise?"),
        'oldpeak': st.number_input("ST Depression", 0.0, 6.2, 1.0, help="Enter your ST depression value"),
        'slope': st.selectbox("Slope of Peak Exercise ST", [0, 1, 2], format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x], help="Select the slope of your peak exercise ST segment"),
        'ca': st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy"),
        'thal': st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Not Available"
        }[x], help="Select your thalassemia type")
    }

    if st.button("Predict"):
        prediction = predict_heart_disease(model, list(user_input.values()))
        result = "Positive for Heart Disease" if prediction == 1 else "No Heart Disease"
        
        # Create a more detailed result display
        st.markdown("### Prediction Results")
        if prediction == 1:
            st.error(f"**Result: Positive for Heart Disease**")
            st.markdown("""
            **Recommendations:**
            - Schedule an appointment with a cardiologist
            - Monitor your blood pressure and heart rate regularly
            - Maintain a heart-healthy diet
            - Exercise regularly under medical supervision
            - Take prescribed medications as directed
            - Consider lifestyle changes:
              * Reduce salt intake
              * Increase physical activity
              * Manage stress levels
              * Quit smoking if applicable
              * Limit alcohol consumption
            """)
        else:
            st.success(f"**Result: No Heart Disease**")
            st.markdown("""
            **Preventive Measures:**
            - Continue regular health check-ups
            - Maintain a healthy lifestyle
            - Exercise regularly
            - Eat a balanced diet
            - Manage stress levels
            - Additional recommendations:
              * Monitor blood pressure regularly
              * Maintain healthy cholesterol levels
              * Stay physically active
              * Get adequate sleep
              * Practice stress management
            """)
        
        # Add disclaimer
        st.info("""
        ⚠️ **Important Disclaimer:** This prediction is based on machine learning algorithms and should not be considered as a definitive medical diagnosis. 
        Always consult with healthcare professionals for proper medical advice and diagnosis.
        """)

# -------------------- EDA Section --------------------
if selected == "EDA":
    st.title("📊 Data Visualization")
    
    # Add dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Heart Disease Cases", len(data[data['target'] == 1]))

    # Add age distribution
    st.subheader("Age Distribution")
    fig = px.histogram(data, x='age', nbins=30, color='target',
                      title='Age Distribution by Heart Disease Status')
    st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".2f", annot_kws={"size": 8})
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    st.write("This may take a moment to load...")
    sns.pairplot(data, hue='target', diag_kind='kde')
    st.pyplot(plt)

    st.subheader("Chest Pain Type Distribution")
    cp_counts = data['cp'].value_counts().reset_index()
    cp_counts.columns = ['Chest Pain Type', 'Count']
    fig = px.bar(cp_counts, x='Chest Pain Type', y='Count', color='Chest Pain Type')
    st.plotly_chart(fig)

    # Add gender distribution
    st.subheader("Gender Distribution")
    gender_counts = data['sex'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    gender_counts['Gender'] = gender_counts['Gender'].map({1: 'Male', 0: 'Female'})
    fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution')
    st.plotly_chart(fig)

# -------------------- Medical History Section --------------------
if selected == "Medical History":
    st.title("📖 Medical History")
    
    # Add search functionality
    search_term = st.text_input("Search in medical records", "")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        age_filter = st.slider("Age Range", int(data['age'].min()), int(data['age'].max()), (int(data['age'].min()), int(data['age'].max())))
    with col2:
        target_filter = st.selectbox("Heart Disease Status", ["All", "Positive", "Negative"])
    
    # Filter data based on user input
    filtered_data = data[
        (data['age'].between(age_filter[0], age_filter[1])) &
        ((target_filter == "All") | 
         (target_filter == "Positive" and data['target'] == 1) |
         (target_filter == "Negative" and data['target'] == 0))
    ]
    
    st.write(f"Showing {len(filtered_data)} records")
    st.write(filtered_data)

# -------------------- Feedback Section --------------------
if selected == "Feedback":
    st.title("💬 Feedback")
    
    # Add rating system
    st.subheader("Rate Our Application")
    rating = st.slider("Rating", 1, 5, 3)
    st.write(f"Your rating: {'⭐' * rating}")
    
    # Add feedback categories
    feedback_category = st.selectbox(
        "Feedback Category",
        ["General", "Prediction Accuracy", "User Interface", "Features", "Suggestions"]
    )
    
    feedback = st.text_area("Provide your feedback here")
    
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback! We appreciate your input.")
            # Here you would typically save the feedback to the database
        else:
            st.warning("Please provide some feedback before submitting.")

# -------------------- Close Connection --------------------
conn.close()

