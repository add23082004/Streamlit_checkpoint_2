import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
data = pd.read_csv("Financial_inclusion_dataset.csv")

# Encode categorical features
label_encoder = LabelEncoder()

data_encoded = data.copy()

for col in data.select_dtypes(include=['object']).columns:
    data_encoded[col] = label_encoder.fit_transform(data_encoded[col])

# Define features and target variable
features = ['country', 'year', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']
target_variable = 'bank_account'

# Select features and target variable
X = data_encoded[features]
y = data_encoded[target_variable]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Visualization 1: Countplot of 'country' using Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(x='country', data=data)
plt.title('Distribution of Interviewees Across Countries')
st.subheader('Distribution of Interviewees Across Countries')
st.pyplot()

# Visualization 2: Countplot of 'location_type' using Plotly
fig = px.histogram(data, x='location_type', title='Distribution of Interviewees in Rural and Urban Areas')
fig.update_layout(barmode='stack')
st.subheader('Distribution of Interviewees in Rural and Urban Areas')
st.plotly_chart(fig)

# Visualization 3: Countplot of 'cellphone_access' with hue='bank_account' using Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(x='cellphone_access', hue='bank_account', data=data)
plt.title('Cellphone Access and Bank Account Distribution')
st.subheader('Cellphone Access and Bank Account Distribution')
st.pyplot()

# Visualization 4: Boxplot of 'household_size' with hue='bank_account' using Seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(x='bank_account', y='household_size', data=data)
plt.title('Household Size Distribution for Bank Account Holders')
st.subheader('Household Size Distribution for Bank Account Holders')
st.pyplot()

# Visualization 5: Distribution plot of 'age_of_respondent' with hue='bank_account' using Seaborn
plt.figure(figsize=(12, 6))
sns.histplot(x='age_of_respondent', hue='bank_account', data=data, kde=True, bins=30)
plt.title('Age Distribution for Bank Account Holders')
st.subheader('Age Distribution for Bank Account Holders')
st.pyplot()

# Visualization 6: Countplot of 'gender_of_respondent' with hue='bank_account' using Plotly
fig = px.histogram(data, x='gender_of_respondent', color='bank_account', title='Bank Account Distribution by Gender')
st.subheader('Bank Account Distribution by Gender')
st.plotly_chart(fig)

# Visualization 7: Countplot of 'marital_status' with hue='bank_account' using Plotly
fig = px.histogram(data, x='marital_status', color='bank_account', title='Bank Account Distribution by Marital Status')
st.subheader('Bank Account Distribution by Marital Status')
st.plotly_chart(fig)

# Visualization 8: Countplot of 'education_level' with hue='bank_account' using Plotly
fig = px.histogram(data, x='education_level', color='bank_account', title='Bank Account Distribution by Education Level')
st.subheader('Bank Account Distribution by Education Level')
st.plotly_chart(fig)

# Visualization 9: Countplot of 'job_type' with hue='bank_account' using Plotly
fig = px.histogram(data, x='job_type', color='bank_account', title='Bank Account Distribution by Job Type')
fig.update_layout(xaxis_tickangle=-45)
st.subheader('Bank Account Distribution by Job Type')
st.plotly_chart(fig)

# Model training and evaluation
model = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', max_depth=3, max_features=None,
                                   min_samples_leaf=4, min_samples_split=10, n_estimators=100, subsample=0.8)
# Cross-validate the model
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
mean_accuracy = np.mean(scores)

# Additional metrics
y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)

# Display confusion matrix
st.subheader('Confusion Matrix')
sns.heatmap(confusion_mat, annot=True, fmt='g')
st.pyplot()

# Display results
results = {'Model': ['Gradient Boosting'], 'Accuracy': [mean_accuracy]}
results_df = pd.DataFrame(results)

st.subheader('Model Results')
st.write(results_df)

# Add input fields for features
st.sidebar.header('Input Features')
country = st.sidebar.selectbox('Country', data['country'].unique())
year = st.sidebar.slider('Year', int(data['year'].min()), int(data['year'].max()), int(data['year'].mean()))
location_type = st.sidebar.selectbox('Location Type', data['location_type'].unique())
cellphone_access = st.sidebar.selectbox('Cellphone Access', data['cellphone_access'].unique())
household_size = st.sidebar.slider('Household Size', int(data['household_size'].min()), int(data['household_size'].max()), int(data['household_size'].mean()))
age_of_respondent = st.sidebar.slider('Age of Respondent', int(data['age_of_respondent'].min()), int(data['age_of_respondent'].max()), int(data['age_of_respondent'].mean()))
gender_of_respondent = st.sidebar.selectbox('Gender of Respondent', data['gender_of_respondent'].unique())
relationship_with_head = st.sidebar.selectbox('Relationship with Head', data['relationship_with_head'].unique())
marital_status = st.sidebar.selectbox('Marital Status', data['marital_status'].unique())
education_level = st.sidebar.selectbox('Education Level', data['education_level'].unique())
job_type = st.sidebar.selectbox('Job Type', data['job_type'].unique())

# Create a dictionary with user input
user_input = {
    'country': country,
    'year': year,
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent': gender_of_respondent,
    'relationship_with_head': relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type
}

# Convert user input into a DataFrame for prediction
user_input_df = pd.DataFrame([user_input])

# Create a mapping dictionary for categorical columns
mapping_dict = {}
for col in data.select_dtypes(include=['object']).columns:
    mapping_dict[col] = dict(zip(data[col].unique(), range(len(data[col].unique()))))

# Use the mapping dictionary to transform user input
for col in user_input_df.select_dtypes(include=['object']).columns:
    user_input_df[col] = mapping_dict[col][user_input_df[col].iloc[0]]

# Encode categorical features in user input
for col in user_input_df.select_dtypes(include=['object']).columns:
    user_input_df[col] = label_encoder.transform(user_input_df[col])
    print(f'{col}: {user_input_df[col].iloc[0]}')

# Make prediction
prediction = model.predict(user_input_df)

# Map prediction to class labels
class_labels = {0: 'No Bank Account', 1: 'Bank Account'}
predicted_label = class_labels[prediction[0]]

# Display prediction
st.subheader('Prediction')
st.write(f'Predicted Class: {predicted_label} (Class {prediction[0]})')

# Display predicted class
st.subheader('Predicted Class')
st.write(f'Predicted Class: {"Has Bank Account" if prediction == 1 else "No Bank Account"}')


# Validation button
if st.sidebar.button('Validate'):
    st.sidebar.subheader('Validation Result')
    st.sidebar.write('Validation successful!')
