import streamlit as st
import pandas as pd
import tensorflow as tf

# Load the trained ANN model
model = tf.keras.models.load_model(r'C:\Users\Raksha.N\Desktop\ANN\Assets\trained_model.h5')

# Define the Streamlit app code
def main():
    st.title('Customer Churn Prediction')

    # Add description and instructions
    st.write("This app predicts customer churn using an Artificial Neural Network (ANN) model.")
    st.write("Enter the required information and click 'Predict' to get the churn prediction.")

    # Create input elements for user input
    credit_score = st.number_input("Credit Score")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age")
    tenure = st.number_input("Tenure")
    balance = st.number_input("Balance")
    num_of_products = st.number_input("Number of Products")
    has_credit_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
    is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
    estimated_salary = st.number_input("Estimated Salary")

    # Perform prediction when the user clicks 'Predict'
    if st.button('Predict'):
        # Create a dictionary with the input values
        input_data = {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_credit_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
            "DummyColumn1": [0],  # Placeholder for missing column
            "DummyColumn2": [0]   
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Convert categorical columns to one-hot encoding
        input_df = pd.get_dummies(input_df, columns=['Geography', 'Gender'])

        # Map 'HasCrCard' and 'IsActiveMember' columns to numerical representations
        input_df["HasCrCard"] = input_df["HasCrCard"].apply(lambda x: 1 if x == "Yes" else 0)
        input_df["IsActiveMember"] = input_df["IsActiveMember"].apply(lambda x: 1 if x == "Yes" else 0)

        # Drop unnecessary columns
        #input_df = input_df.drop(columns=['Geography', 'Gender'])


        # Convert input data to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(input_df.values, dtype=tf.float32)

        # Reshape the input tensor to match the expected shape
        input_tensor = tf.reshape(input_tensor, (1, -1))

        # Make predictions using the loaded model
        prediction = model.predict(input_tensor)
        churn_probability = prediction[0][0]

        # Display the churn prediction result
        if churn_probability >= 0.5:
            st.write("Churn Probability:", churn_probability)
            st.warning("The customer is predicted to not churn.")
        else:
            st.write("Churn Probability:", churn_probability)
            st.success("The customer is predicted to churn.")

if __name__ == '__main__':
    main()
