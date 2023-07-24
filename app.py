import streamlit as st
from PIL import Image
import pandas as pd
import pickle

# Reverse mapping dictionaries
mapping_gender = {'Male':1, 'Female':0}
mapping_seniorcitizen = {'Yes':1, 'No':0}
mapping_partner = {'Yes':1, 'No':0}
mapping_phoneservice = {'Yes':1, 'No':0}
mapping_paperlessbilling = {'Yes':1, 'No':0}
mapping_dependents = {'Yes':1, 'No':0}

mapping_MultipleLines = {'No phone service': 0, 'No': 1, 'Yes': 2}
mapping_InternetService = {'DSL': 0, 'No': 1, 'Fiber optic': 2}
mapping_OnlineSecurity = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_OnlineBackup = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_DeviceProtection = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_TechSupport = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_StreamingTV = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_StreamingMovies = {'No': 0, 'Yes': 1, 'No internet service': 2}
mapping_Contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
mapping_PaymentMethod = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2,'Credit card (automatic)':3}

def main():
    # image = Image.open(r"C:\Users\asus\Downloads\728931-405442638.jpg")
    image2 = Image.open(r"C:\Users\asus\Downloads\logo.ico")
    # st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Folder CSV"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    

    with open(r'task-1\logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)



    if add_selectbox == 'Online':
        gender = st.selectbox('Gender:', ['Male', 'Female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', ['Yes', 'No'])
        partner = st.selectbox('Customer has a partner:', ['Yes', 'No'])
        dependents = st.selectbox('Customer has dependents:', ['Yes', 'No'])
        phoneservice = st.selectbox('Customer has phoneservice:', ['Yes', 'No'])
        multiplelines = st.selectbox('Customer has multiplelines:', ['No phone service','No','Yes'])
        internetservice = st.selectbox('Customer has internetservice:', ['DSL','No','Fiber optic'])
        onlinesecurity = st.selectbox('Customer has onlinesecurity:',['No','Yes','No internet service'])
        onlinebackup = st.selectbox('Customer has onlinebackup:', ['No','Yes','No internet service'])
        deviceprotection = st.selectbox('Customer has deviceprotection:', ['No','Yes','No internet service'])
        techsupport = st.selectbox('Customer has techsupport:', ['No','Yes','No internet service'])
        streamingtv = st.selectbox('Customer has streamingtv:', ['No','Yes','No internet service'])
        streamingmovies = st.selectbox('Customer has streamingmovies:', ['No','Yes','No internet service'])
        contract = st.selectbox('Customer has a contract:',  ['Month-to-month','One year','Two year'])
        paperlessbilling = st.selectbox('Customer has a paperlessbilling:', ['Yes', 'No'])
        paymentmethod = st.selectbox('Payment Option:', ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=300, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges

        # Reverse mapping for user-selected values
        gender = mapping_gender[gender]
        

        multiplelines = mapping_MultipleLines[multiplelines]
        seniorcitizen = mapping_seniorcitizen[seniorcitizen]
        partner = mapping_partner[partner]
        phoneservice = mapping_phoneservice[phoneservice]
        paperlessbilling = mapping_paperlessbilling[paperlessbilling]
        dependents = mapping_dependents[dependents]

        internetservice = mapping_InternetService[internetservice]
        onlinesecurity = mapping_OnlineSecurity[onlinesecurity]
        onlinebackup = mapping_OnlineBackup[onlinebackup]
        deviceprotection = mapping_DeviceProtection[deviceprotection]
        techsupport = mapping_TechSupport[techsupport]
        streamingtv = mapping_StreamingTV[streamingtv]
        streamingmovies = mapping_StreamingMovies[streamingmovies]
        contract = mapping_Contract[contract]
        paymentmethod = mapping_PaymentMethod[paymentmethod]

        input_dict = {
            "SeniorCitizen": seniorcitizen,
            "tenure": tenure,
            "MultipleLines": multiplelines,
            "InternetService": internetservice,
            "OnlineSecurity": onlinesecurity,
            "OnlineBackup": onlinebackup,       
            "DeviceProtection": deviceprotection,
            "TechSupport": techsupport,
            "StreamingTV": streamingtv,
            "StreamingMovies": streamingmovies,
            "Contract": contract,
            "PaymentMethod": paymentmethod, 
            "MonthlyCharges": monthlycharges,
            "TotalCharges": totalcharges,
            "gender_Male": gender,
            "Partner_Yes": partner,
            "Dependents_Yes": dependents,
            "PhoneService_Yes": phoneservice,
            "PaperlessBilling_Yes": paperlessbilling,
        }

        if st.button("Predict"):
            input_df = pd.DataFrame([input_dict])
            churn_probability = model.predict_proba(input_df)[:, 1]
            churn = churn_probability >= 0.5

            st.success('Churn: {0}, Risk Score: {1:.2f}%'.format(churn[0], churn_probability[0] * 100))


    if add_selectbox == "Folder CSV":
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            mo = 1

            if mo == 1:
                data['TotalCharges'] = data['tenure'] * data['MonthlyCharges']

                X_batch = data
                churn_probabilities = model.predict_proba(X_batch)[:, 1]
                churn_predictions = churn_probabilities >= 0.5

                data['Churn'] = churn_predictions
                data['Churn_Probability'] = churn_probabilities

                st.write(data)





if __name__ == '__main__':
    main()






                # data['SeniorCitizen'] = data['SeniorCitizen'].map(mapping_seniorcitizen)
                # data['MultipleLines'] = data['MultipleLines'].map(mapping_MultipleLines)
                # data['InternetService'] = data['InternetService'].map(mapping_InternetService)
                # data['OnlineSecurity'] = data['OnlineSecurity'].map(mapping_OnlineSecurity)
                # data['OnlineBackup'] = data['OnlineBackup'].map(mapping_OnlineBackup)
                # data['DeviceProtection'] = data['DeviceProtection'].map(mapping_DeviceProtection)
                # data['TechSupport'] = data['TechSupport'].map(mapping_TechSupport)
                # data['StreamingTV'] = data['StreamingTV'].map(mapping_StreamingTV)
                # data['StreamingMovies'] = data['StreamingMovies'].map(mapping_StreamingMovies)
                # data['Contract'] = data['Contract'].map(mapping_Contract)
                # data['PaymentMethod'] = data['PaymentMethod'].map(mapping_PaymentMethod)
                # data['gender_Male'] = data['gender_Male'].map(mapping_gender)
                # data['Partner_Yes'] = data['Partner_Yes'].map(mapping_partner)
                # data['Dependents_Yes'] = data['Dependents_Yes'].map(mapping_dependents)
                # data['PhoneService_Yes'] = data['PhoneService_Yes'].map(mapping_phoneservice)
                # data['PaperlessBilling_Yes'] = data['PaperlessBilling_Yes'].map(mapping_paperlessbilling)

































