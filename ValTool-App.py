import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import base64
import requests

# Define the available options for country and industry
countries = {
    'United States': 'North America', 'Canada': 'North America', 'Greenland': 'North America',
    'France': 'Western Europe', 'Germany': 'Western Europe', 'Netherlands': 'Western Europe',
    'Belgium': 'Western Europe', 'Luxembourg': 'Western Europe', 'Switzerland': 'Western Europe',
    'Austria': 'Western Europe', 'Liechtenstein': 'Western Europe', 'Monaco': 'Western Europe',
    'United Kingdom': 'UK', 'Jersey': 'UK', 'Guernsey': 'UK', 'Isle of Man': 'UK', 'Ireland': 'UK',
    'Sweden': 'Nordics', 'Norway': 'Nordics', 'Denmark': 'Nordics', 'Finland': 'Nordics', 'Iceland': 'Nordics',
    'Poland': 'Eastern Europe', 'Bosnia and Herzegovina': 'Eastern Europe', 'Lithuania': 'Eastern Europe',
    'Bulgaria': 'Eastern Europe', 'Russia': 'Eastern Europe', 'Estonia': 'Eastern Europe',
    'Latvia': 'Eastern Europe', 'Hungary': 'Eastern Europe', 'Romania': 'Eastern Europe', 'Ukraine': 'Eastern Europe',
    'Moldova': 'Eastern Europe', 'Serbia': 'Eastern Europe', 'Slovenia': 'Eastern Europe',
    'North Macedonia': 'Eastern Europe', 'Montenegro': 'Eastern Europe', 'Slovakia': 'Eastern Europe',
    'Czech Republic': 'Eastern Europe', 'Portugal': 'Southern Europe', 'Italy': 'Southern Europe',
    'Spain': 'Southern Europe', 'Greece': 'Southern Europe', 'Croatia': 'Southern Europe',
    'Cyprus': 'Southern Europe', 'Malta': 'Southern Europe', 'Gibraltar': 'Southern Europe', 'Turkey': 'Southern Europe'
}
industries = [
    'Consumer Products and Services', 'Consumer Staples', 'Energy and Power', 'Financials',
    'Healthcare', 'High Technology', 'Industrials', 'Materials', 'Media and Entertainment',
    'Real Estate', 'Retail', 'Telecommunications'
]

# Download the model from Dropbox
@st.cache_data(show_spinner=False)
def download_model_from_dropbox():
    url = 'https://www.dropbox.com/scl/fi/0yixy98hccpu309cpojez/model.pkl?rlkey=ldr8mm08fzr49jpctmk6pckv3&st=e8ueabx6&dl=0'
    local_model_path = 'model/model.pkl'
    
    # Create local model directory if it doesn't exist
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    response = requests.get(url)
    with open(local_model_path, 'wb') as f:
        f.write(response.content)

#Load the trained model
def load_model():
    local_model_path = 'model/model.pkl'
    
    # Check if model is already downloaded
    if not os.path.exists(local_model_path):
        download_model_from_dropbox()

    return joblib.load(local_model_path)

model = load_model()

# Add an image at the top
def load_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image_base64 = load_image(os.path.join('images', 'finance.png'))
st.markdown(f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{image_base64}' style='width: 25%; height: auto;'>
    </div>
""", unsafe_allow_html=True)

# Add the title and directions at the top
st.markdown("<h1 style='text-align: center;'>AI-Powered Valuation Tool</h1>", unsafe_allow_html=True)
st.markdown("<h4>Enter information about the company below:</h4>", unsafe_allow_html=True)

# Create the UI components
# Revenues and EBITDA input (will be log-transformed later)
revenues = st.number_input('Revenues (in millions of €)', min_value=0.0, format="%.2f")
ebitda = st.number_input('EBITDA (in millions of €)', min_value=0.0, format="%.2f")
# Company status (public or private, will be converted to binary)
company_status = st.radio('Company Status', ('Public', 'Private'))
# Country selection (dropdown list)
selected_country = st.selectbox('Country', sorted(countries.keys()))
# Industry selection (dropdown list)
selected_industry = st.selectbox('Industry', industries)
market_condition = st.radio('Market Condition', ('Low Market', 'Normal Market', 'High Market'))

# Submit button
if st.button('Submit'):
    #Check if revenues or EBITDA are 0
    if revenues == 0 or ebitda == 0:
        st.error("Revenues and EBITDA values must be greater than 0. Please enter valid values.")
        st.stop()
    else:
        # Log transformations for revenues and EBITDA
        log_revenues = np.log(revenues)
        log_ebitda = np.log(ebitda) if ebitda > 0 else -np.log(abs(ebitda))
    # One-hot encode the market condition
    if market_condition == 'Low Market':
        Market_HM, Market_NM, Market_LM = 0, 0, 1
    elif market_condition == 'Normal Market':
        Market_HM, Market_NM, Market_LM = 0, 1, 0
    else:
        Market_HM, Market_NM, Market_LM = 1, 0, 0
    # Convert company status to binary (1 for public, 0 for private)
    company_status_public = 1 if company_status == 'Public' else 0
    # Create region one-hot encoded variables
    region = countries[selected_country]
    regions = {
        'Target_Region_Eastern Europe': 1 if region == 'Eastern Europe' else 0,
        'Target_Region_Nordics': 1 if region == 'Nordics' else 0,
        'Target_Region_North America': 1 if region == 'North America' else 0,
        'Target_Region_Southern Europe': 1 if region == 'Southern Europe' else 0,
        'Target_Region_UK': 1 if region == 'UK' else 0,
        'Target_Region_Western Europe': 1 if region == 'Western Europe' else 0
    }
    # Create industry one-hot encoded variables
    industry_macro = f'Target_Industry_Macro_{selected_industry}'
    industries_one_hot = {f'Target_Industry_Macro_{industry}': 1 if industry == selected_industry else 0 for industry in industries}
    # Combine all inputs into a DataFrame (matching the X_new format)
    X_new = pd.DataFrame({
        'log_Target_Revenues': [log_revenues],
        'log_Target_EBITDA': [log_ebitda],
        'Market_HM': [Market_HM],
        'Market_NM': [Market_NM],
        'Market_LM': [Market_LM],
        'Target_Status_Public': [company_status_public],
        **regions,
        **industries_one_hot
    })
    
    # Use the loaded model to make a prediction
    prediction = model.predict(X_new)
       
    #Display the prediction
    st.markdown(f"<h2>Predicted Company Value in millions: €{np.exp(prediction[0]):,.2f}</h3>", unsafe_allow_html=True)
