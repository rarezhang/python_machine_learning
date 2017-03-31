from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np


# import HashingVectorizer from local dir
from vectorizer import vect

# import update function from local dir
from update import update_model

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 
                 'pkl_objects/classifier.pkl'), 'rb'))
                 
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba
    
def train(document, y):
    """
    update the classifier given that a document and a class label are provided 
    """
    X = vect.transform([document])
    clf.partial_fit(X, [y])
    
    
def sqlite_entry(path, document, y):
    """
    store a submitted movie review in the database 
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()
    
    
    
app = Flask(__name__)
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
                                
                                
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)
    
    
@app.route('/results', methods=['POST'])
def results():
    """
    fetch the contents of the submitted web form and pass it on to classifier to predict the sentiment of the movie classifier 
    """
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)
    
    
@app.route('/thanks', methods=['POST'])
def feedback():
    """
    fetch the predicted class label from the results.html template if a user clicked on the Correct or Incorrect feedback button 
    transforms the predicted sentiment back into an integer class label --> will be used to update the classifier via the train fun 
    """
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')
    
    
if __name__ == '__main__':
    app.run(debug=True)
    # update_model(filepath=db, model=clf, batch_size=10000)