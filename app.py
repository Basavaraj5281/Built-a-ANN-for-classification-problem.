import pandas as pd 
import tensorflow as tf 
import streamlit as st
import numpy as np 
import pickle


# Load model 
model=tf.keras.models.load_model("model.h5")


# load OheHotEncoder for geogrphy file 
with open("OHE_geography.pkl","rb") as file:
    OHE_geography=pickle.load(file)

# Gender label encoder 
with open("Gender_encoder.pkl","rb") as file:
    Gender_encoder=pickle.load(file)
# StandardScaling model
with open("scaler.pkl","rb")  as file:
    scaler=pickle.load(file)

st.title("Churn Prediction")
georaphy=st.selectbox("Georaphy",OHE_geography.categories_[0])
gender=st.selectbox("Gender",Gender_encoder.classes_)
age=st.slider("Age",18,95)
CreditScore=st.number_input("CreditScore")
balance=st.number_input("Balance")
estimated_salary=st.number_input("Estimated salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.selectbox("Number of products",[1,2,3,4])
has_cr_card=st.selectbox("has credict card ",[0,1])
is_active_member=st.selectbox("Is active member",[0,1])


input_data=pd.DataFrame({
    
    "CreditScore":[CreditScore],
    "Geography"	:[georaphy],
    "Gender":[Gender_encoder.transform([gender])[0]],
    "Age":[age],	
    "Tenure":[tenure],	
    "Balance":[balance],	
    "NumOfProducts":[num_of_products],	
    "HasCrCard":[has_cr_card],	
    "IsActiveMember":[is_active_member],	
    "EstimatedSalary":[estimated_salary]
})




## OHE Encode the Geogrphy 
geo_encoded=OHE_geography.transform([[georaphy]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=OHE_geography.get_feature_names_out(["Geography"]))
##Label encode the gender 


#Concate the data
input_data=pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)
# Standard scaling the input data 
input_data_scaled=scaler.transform(input_data)
#Now predict the output

prediction=model.predict(input_data_scaled)

prediction_prob=prediction[0][0]

if prediction_prob>0.5:
    st.write("The coustomer is likely to churn ")
else:
    st.write("The coustomer is not likely to churn ")

st.write(prediction_prob)