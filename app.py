#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask


# In[ ]:


app = Flask(__name__)


# In[ ]:


from flask import render_template, request
import joblib

@app.route("/", methods = ["GET","POST"])

def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loanamount = request.form.get("loan")
        
        income = float(income)
        age = float(age)
        loanamount = float(loanamount)
        print(income, age, loanamount)
        
        model1 = joblib.load("CCD_DT")
        pred1 = model1.predict([[income, age, loanamount]])
        str1 = "The score of credit card default based on decision tree is" + str(pred1)
        
        model2 = joblib.load("CCD_Reg")
        pred2 = model2.predict([[income, age, loanamount]])
        str2 = "The score of credit card default based on regression is" + str(pred2)
        
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[income, age, loanamount]])
        str3 = "The score of credit card default based on neural network is" + str(pred3)
        
        model4 = joblib.load("CCD_RF")
        pred4 = model4.predict([[income, age, loanamount]])
        str4 = "The score of credit card default based on random forest is" + str(pred4)
        
        model5 = joblib.load("CCD_GB")
        pred5 = model5.predict([[income, age, loanamount]])
        str5 = "The score of credit card default based on gradient boosting is" + str(pred5)
        
        return(render_template("index.html", result1 = str1,result2 = str2,result3 = str3,result4 = str4,result5 = str5))
    
    else:
        return(render_template("index.html", result1 = "2",result2 = "2",result3 = "2",result4 = "2",result5 = "2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




