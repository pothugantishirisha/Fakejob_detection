import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import db_connect,user_reg,user_loginact
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request
import joblib
import os
import numpy as np
import pickle
import time
import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
 
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json, nltk
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import re
app = Flask(__name__, static_folder='static')
global res

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route("/")
def FUN_root():
    return render_template("index.html")

@app.route("/index.html")
def logout():
    return render_template("index.html")
@app.route("/userhome.html")
def uhome():
    return render_template("userhome.html")

@app.route("/register.html")
def reg():
    return render_template("register.html")

@app.route("/login.html")
def login():
    return render_template("login.html")

@app.route("/upload.html")
def up():
    return render_template("upload.html")

# -------------------------------------------register-------------------------------------------------------
@app.route("/regact", methods = ['GET','POST'])
def registeract():
   if request.method == 'POST':    
      id="0"
      status = user_reg(id,request.form['username'],request.form['password'],request.form['email'],request.form['mobile'],request.form['address'])
      if status == 1:
       return render_template("login.html",m1="sucess")
      else:
       return render_template("register.html",m1="failed")
#--------------------------------------------Login-----------------------------------------------------
@app.route("/loginact", methods=['GET', 'POST'])
def useract():
    if request.method == 'POST':
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']                             
            return render_template("userhome.html", m1="sucess")
        else:
            return render_template("login.html", m1="Login Failed")
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet



def process_tweet(tweet):
    with open('assets/contractions.json', 'r') as f:
        contractions_dict = json.load(f)
    contractions = contractions_dict['contractions']    
    tweet = tweet.lower()                                             # Lowercases the string
    tweet = re.sub('@[^\s]+', '', tweet)                              # Removes usernames
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)   # Remove URLs
    tweet = re.sub(r"\d+", " ", str(tweet))                           # Removes all digits
    tweet = re.sub('&quot;'," ", tweet)                               # Remove (&quot;) 
    tweet = emoji(tweet)                                              # Replaces Emojis
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    for word in tweet.split():
        if word.lower() in contractions:
            tweet = tweet.replace(word, contractions[word.lower()])   # Replaces contractions
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)                         # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))                           # Replaces double spaces with single space    
    return tweet
#-------------------------------------------Upload Image----------------------------------
@app.route('/result', methods=['POST', 'GET'])
def result():
     
    jobid = str(request.form['jobid'])
    title = str(request.form['title'])
    location = str(request.form['location'])
    department = str(request.form['department'])
    salaryrange = str(request.form['salaryrange'])
    companyprofile = str(request.form['companyprofile'])
    description = str(request.form['description'])
    requirements = str(request.form['requirements'])
    benefits = str(request.form['benefits'])
    telecommuting = str(request.form['telecommuting'])
    hascompanylogo = str(request.form['hascompanylogo'])
    hasquestions = str(request.form['hasquestions'])
    employmenttype = str(request.form['employmenttype'])
    requiredexperience = str(request.form['requiredexperience'])
    requirededucation = str(request.form['requirededucation'])
    industry = str(request.form['industry'])
    function = str(request.form['function'])
    fraudulent = 0
    col_names =  ['jobid','title','location','department','salaryrange','companyprofile','description','requirements','benefits','telecommuting','hascompanylogo','hasquestions','employmenttype','requiredexperience','requirededucation','industry','function','fraudulent']
    jobprofile=title
    total_data = pd.read_csv("dataset/train.csv", encoding="ISO-8859-1")

   
    pd.set_option('display.max_colwidth', -1)

    job = total_data.columns.values[1]
    sentiment = total_data.columns.values[17]    
    total_data['processed_job'] = np.vectorize(process_tweet)(total_data[job])
    count_vectorizer = CountVectorizer(ngram_range=(1,2))
    final_vectorized_data = count_vectorizer.fit_transform(total_data['processed_job'])     # Unigram and Bigram
    ytb_model = open("fakejob.pkl","rb")
    new_model = pickle.load(ytb_model)
    comment2 = [jobprofile]        
    check = count_vectorizer.transform(comment2).toarray()
    predicted_naive = new_model.predict(check)   
    print(predicted_naive)
    if predicted_naive == 0:
        print("Genuine Job")
        res="Genuine Job"
    elif predicted_naive == 1:
        print("Fake Job Detected")
        res="Fake Job Detected"
    return render_template('viewdata.html', first=str(res),uname=session['username'])      
    
    
# ----------------------------------------------Update Item------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000,use_reloader=False)