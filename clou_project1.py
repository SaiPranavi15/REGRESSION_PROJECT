import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Title & Description
st.title("Simple Linear Regression - Salary Prediction")
st.markdown("This app helps you visualize and predict salary based on years of experience using linear regression.")

# Step 1: Load and Display Dataset
st.header("Step 1: Load and Display Data")

upload_file = st.file_uploader("Upload your CSV file", type="csv")

if upload_file is not None:
    data = pd.read_csv(upload_file)
else:
    # Fallback to embedded data if file not uploaded
    from io import StringIO

    csv_data = StringIO("""YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
2.0,43525.00
2.2,39891.00
2.9,56642.00
3.0,60150.00
3.2,54445.00
3.2,64445.00
3.7,57189.00
3.9,63218.00
4.0,55794.00
4.0,56957.00
4.1,57081.00
4.5,61111.00
4.9,67938.00
5.1,66029.00
5.3,83088.00
5.9,81363.00
6.0,93940.00
6.8,91738.00
7.1,98273.00
7.9,101302.00
8.2,113812.00
8.7,109431.00
9.0,105582.00
9.5,116969.00
9.6,112635.00
10.3,122391.00
10.5,121872.00""")
    data = pd.read_csv(csv_data)

st.write("Preview of Dataset")
st.dataframe(data)

# Step 2: Scatter Plot - Experience vs Salary
st.header("Step 2: Visualize Salary vs. Experience")

fig, ax = plt.subplots()
ax.scatter(data["YearsExperience"], data["Salary"], color="blue")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")
st.pyplot(fig)

# Step 3: Train the Linear Regression Model
st.header("Step 3: Train the Model")

x = data[["YearsExperience"]]
y = data[["Salary"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

st.success("Model trained successfully!")

# Step 4: Plot the Regression Line
st.header("Step 4: Visualize Regression Line")

fig2, ax2 = plt.subplots()
ax2.scatter(x, y, color="blue", label="Actual Data")
ax2.plot(x, model.predict(x), color="red", label="Regression Line")
ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Salary")
ax2.set_title("Regression Line")
ax2.legend()
st.pyplot(fig2)

# Step 5: Predict Salary from User Input
st.header("Step 5: Predict Salary")

experience = st.number_input("Enter years of experience:", min_value=0.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    predicted_salary = float(prediction[0][0])
    st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")

# Step 6: Display Model Performance
st.header("Step 6: Model Performance")

r2 = r2_score(y_test, model.predict(x_test))
st.metric(label="Model R² Score", value=f"{r2:.2f}")
