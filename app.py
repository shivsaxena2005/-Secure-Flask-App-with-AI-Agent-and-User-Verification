from flask import Flask, request, render_template, session, url_for, redirect, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import json
import random
from werkzeug.security import generate_password_hash, check_password_hash  # Secure password handling
import os,joblib,numpy as np,pandas as pd
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generates a random key each time the app restarts

# Load configuration from JSON
with open('templates/config.json', 'r') as f:
    params = json.load(f)['params']

# ✅ Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

# ✅ Database Configuration
app.secret_key = "loginform"

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql://root:Rajat%40123@localhost/practise')

#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Rajat%40123@localhost/practise'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ✅ Database Model
class Contact(db.Model):
    __tablename__ = 'first'
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Increased length for hash
    email = db.Column(db.String(255), unique=True, nullable=False)
    branch = db.Column(db.String(10), nullable=False)
    address = db.Column(db.String(20), nullable=False)

# ✅ Home Page
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Register User
@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        branch = request.form.get('branch')
        password = request.form.get('password')
        address = request.form.get('address')
       # if()
        if not all([name, email, branch, password, address]):
            flash("⚠️ Error: All fields are required!", "danger")
            return render_template('register.html')

        # ✅ Hash the password before storing
        hashed_password = generate_password_hash(password)

        # ✅ Save user data
        entry = Contact(name=name, password=hashed_password, email=email, branch=branch, address=address)
        db.session.add(entry)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# ✅ Login User
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = Contact.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['email'] = user.email
            return render_template('ml_files.html')
        return(render_template('wrong.html'))
        flash("Invalid email or password", "danger")
        return redirect(url_for('login'))
    print("Error")
    return render_template('login.html')

# ✅ Forgot Password - Step 1: Enter Email
@app.route('/forget_pass', methods=['POST', 'GET'])
def forget_pass():
    if request.method == 'POST':
        email = request.form.get('email')

        if not email:
            flash("⚠️ Please enter your email!", "danger")
            return redirect(url_for('forget_pass'))

        user = Contact.query.filter_by(email=email).first()
        if user:
            otp = str(random.randint(100000, 999999))  # Generate 6-digit OTP
            session['otp'] = otp
            session['email'] = email

            # ✅ Send OTP Email
            msg = Message('OTP Verification Mail',
                          sender=app.config['MAIL_USERNAME'],
                          recipients=[email])
            msg.body = f"Your OTP is: {otp}"
            mail.send(msg)

            flash("✅ OTP has been sent to your email!", "success")
            return render_template('otp_validation.html')  # Redirect to OTP page

        flash("⚠️ Email not found!", "danger")
    print("error")
    return render_template('forget_pass.html')

# ✅ OTP Verification - Step 2
@app.route('/otp_check', methods=['POST', 'GET'])
def otp_check():
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        stored_otp = session.get('otp')

        if stored_otp and entered_otp == stored_otp:
            session.pop('otp')  # Remove OTP after use
            return (render_template('pass_change.html'))  # Redirect to password change page

        flash("❌ Invalid OTP. Please try again.", "danger")
        return redirect(url_for('otp_check'))

    return render_template('otp_validation.html', email=session.get('email'))

# ✅ Change Password - Step 3
@app.route('/pass_change', methods=['POST', 'GET'])
def pass_change():
    if request.method == 'POST':
        new_password = request.form.get('password')

        if 'email' in session:
            email = session['email']
            user = Contact.query.filter_by(email=email).first()

            if user:
                user.password = generate_password_hash(new_password)  # ✅ Secure password storage
                db.session.commit()
                session.pop('email')  # Remove email from session
                flash("✅ Password updated successfully!", "success")
                return redirect(url_for('login'))

            flash("⚠️ User not found!", "danger")

        flash("❌ Unauthorized access!", "danger")
        return redirect(url_for('forget_pass'))

    return render_template('pass_change.html')

@app.route('/knn',methods=['POST','GET'])
def knn():
    if(request.method=='POST'):
        return(render_template('knn.html'))

@app.route('/naive',methods=['POST','GET'])
def naive():
    if(request.method=='POST'):
        return(render_template('naive.html'))

@app.route('/svm',methods=['POST','GET'])
def svm():
    if(request.method=='POST'):
        return(render_template('svm.html'))


@app.route('/linear_regression',methods=['POST','GET'])
def linear_regression():
    if(request.method=='POST'):
        return(render_template('linear_regression.html'))
    
@app.route('/decision_tree',methods=['POST','GET'])
def decision_tree():
    if(request.method=='POST'):
        return(render_template('decision_tree.html'))    


@app.route('/linear_model',methods=['POST','GET'])

def model1():
    if(request.method =='POST'):
        bed = request.form.get('bed')
        bath = request.form.get('bath')
        area = request.form.get('area')
        stories = request.form.get('stories')
        parking = request.form.get('parking')
        Y = (167809.788 * float(bed)) + (1133740.16 * float(bath)) + (331.115495 * float(area)) + (547939.81 * float(stories)) + (377596.289 * float(parking)) - 145734.48945597932
        return(render_template('result.html',Y=Y))
    return(render_template('model_1.html'))
    #[['bedrooms','bathrooms','area','stories','parking']]

@app.route('/knn_2',methods=['POST','GET'])
def knn_new():
    if(request.method=='POST'):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

# 🔹 Step 1: Load the datasetC:\Users\hp\Desktop\project_1\static\KNNAlgorithmDataset.csv
        df = pd.read_csv('static/KNNAlgorithmDataset.csv')

# 🔹 Step 2: Select features and target variable
        X = df[['radius_worst', 'perimeter_worst', 'area_worst', 'concave points_worst', 'concavity_mean']]
        y = df['diagnosis']  # Target column (M or B)

# 🔹 Step 3: Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 4: Initialize and train KNN model
        model = KNeighborsClassifier(n_neighbors=5)  # Using k=5
        model.fit(X_train,y_train)
        print("hello")
# 🔹 Step 5: Make predictions on test data
#y_pred = model.predict(X_test)

# 🔹 Step 6: Evaluate model accuracy
# 🔹 Step 7: Function to predict diagnosis from user input
        try:
        # Get user input
            radius_worst = float(request.form.get('radius_worst'))
            perimeter_worst = float(request.form.get('perimeter_worst'))
            area_worst = float(request.form.get('area_worst'))
            concave_points_worst = float(request.form.get('concave_points_worst'))
            concavity_mean = float(request.form.get('concavity_mean'))

        # Convert input into NumPy array
            user_input = np.array([[radius_worst, perimeter_worst, area_worst, concave_points_worst, concavity_mean]])

        # Make prediction
            prediction = model.predict(user_input)[0]
            print(f"\n🔹 Predicted Diagnosis: {prediction} ({'Malignant' if prediction == 'M' else 'Benign'})\n")
            return(render_template('result_2.html',prediction = 'Malignant' if prediction == 'M' else 'Benign'))
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")

# 🔹 Step 8: Call the function for user input prediction
      #  predict_diagnosis()
    return(render_template('model_2.html'))

@app.route('/decision',methods=['GET','POST'])
def home():
    if(request.method=='POST'):
        model1 = joblib.load('static/updated_model.pkl')
        new = pd.DataFrame({
            'Age': [float(request.form.get('Age'))], 
            'Gender': [float(request.form.get('Gender'))], 
            'AnnualSalary': [float(request.form.get('AnnualSalary'))]
        })
        print(model1.feature_names_in_)  # Shows feature names expected by the model

        prediction = model1.predict(new)
        print("Prediction:", prediction[0])
        if(prediction[0]==1):
            m='Yes'
        else:
            m='NO'    
        return(render_template('result_3.html',m=m))
    return(render_template('model_3.html'))    

@app.route('/naive_',methods=['GET','POST'])
def nai():
    if(request.method == 'POST'):  
        model = joblib.load('static\naive_bayes_diabetes.pkl')
        # Example new patient data
        new_data = pd.DataFrame({
            'glucose': [float(request.form.get('glucose', 100))],  # Default 100 if missing
            'bloodpressure': [float(request.form.get('bloodpressure', 70))]  # Default 70 if missing
        })

# Predict diabetes
        prediction = model.predict(new_data)
        print("Diabetes Prediction:", "Yes" if prediction[0] == 1 else "No")

        prediction = model.predict(new_data)
        print("Diabetes Prediction:", "Yes" if prediction[0] == 1 else "No")
        return(render_template('result_4.html',m="Yes" if prediction[0] == 1 else "No"))
    return(render_template('model_4.html'))
if __name__ == '__main__':
    app.run(debug=True)