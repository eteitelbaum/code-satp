# Streamlit for creating web apps
import streamlit as st

# Web scraping
import requests
from bs4 import BeautifulSoup
import re

# Data manipulation
import pandas as pd
import numpy as np

# NLP with Transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Google Sheets API
import gspread
from google.oauth2.service_account import Credentials

# Utility modules
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       save_to_google_sheets
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Define service account credentials
# service_account_info = {
#     "type": "service_account",
#     "project_id": "satp-data-colab-to-sheets",
#     "private_key_id": userdata.get('private_key_id'),
#     "private_key": userdata.get('private_key').replace('\\n', '\n'),
#     "client_email": userdata.get('client_email'),
#     "client_id": userdata.get('client_id'),
#     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#     "token_uri": "https://oauth2.googleapis.com/token",
#     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
#     "client_x509_cert_url": userdata.get('client_x509_cert_url'),
#     "universe_domain": "googleapis.com"
# }


service_account_info = {
    "type": "service_account",
    "project_id": "satp-data-colab-to-sheets",
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": "googleapis.com"
}

def save_to_google_sheets(data, spreadsheet_name, sheet_name):
    # Set up Google Sheets API credentials
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)

    # Open the spreadsheet and worksheet
    try:
        sheet = client.open(spreadsheet_name).worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # Create a new worksheet if it doesn't exist
        spreadsheet = client.open(spreadsheet_name)
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=data.shape[1])

        # Insert column headers dynamically
        sheet.append_row(list(data.columns))
        print(f"Created new sheet '{sheet_name}' and added column headers.")

    existing_data = pd.DataFrame(sheet.get_all_records())


    # Handle the case where existing_data is empty
    if existing_data.empty:
        # If no data exists in the sheet, upload the entire dataset
        new_rows = data
        sheet.append_row(list(data.columns))
    else:
        if 'Incident_Number' in existing_data.columns and 'Incident_Number' in data.columns:
            # Find new rows based on incident number
            existing_incident_numbers = set(existing_data['Incident_Number'])
            new_rows = data[~data['Incident_Number'].isin(existing_incident_numbers)]
        else:
            new_rows = None

    if new_rows is not None and not new_rows.empty:
        # Replace inf and NaN with strings before uploading
        new_rows = new_rows.replace([np.inf, -np.inf], np.nan) # replace inf with NaN
        new_rows = new_rows.fillna('') # replace NaN with empty string
        # Prepare new data for batch update
        batch_data = new_rows.values.tolist()  # Convert DataFrame to a list of lists
        # Insert new rows
        sheet.append_rows(batch_data)
        print(f"Uploaded {len(new_rows)} new rows to Google Sheet {sheet_name}.")
    else:
        print(f"No new incidents found to upload for the sheet {sheet_name}.")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       infer_perpetrator
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load the saved model and tokenizer
perpetrator_model_path = "perpetrator/distilBert"  # Update with your actual path
perpetrator_model = AutoModelForSequenceClassification.from_pretrained(perpetrator_model_path)
perpetrator_tokenizer = AutoTokenizer.from_pretrained(perpetrator_model_path)

perpetrator_model.to(device)

def infer_perpetrator(summary):
    inputs = perpetrator_tokenizer(summary, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = perpetrator_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0:'Security', 1:'Maoist', 2:'Unknown'}

    predicted_perpetrator = label_map.get(predicted_class, "Unknown")
    perpetrator = {
        'perpetrator': predicted_perpetrator
    }
    return perpetrator

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       inference_action_type
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load the saved model and tokenizer
action_model_path = "action_type/distilbert_model"
action_tokenizer = AutoTokenizer.from_pretrained(action_model_path)
action_model = AutoModelForSequenceClassification.from_pretrained(action_model_path)

action_model.to(device)


def inference_action_type(summary):
    """
    Performs inference on an incident summary to predict action types.

    Args:
        summary: The incident summary text.

    Returns:
        A dictionary with action type labels as keys and their predicted probabilities (0 or 1) as values.
    """

    # Tokenize the input summary
    inputs = action_tokenizer(summary, padding=True, truncation=True, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = action_model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # Get probabilities using sigmoid

    # Convert probabilities to binary predictions (0 or 1) using threshold
    threshold = 0.5
    predictions = (probs > threshold).squeeze().cpu().numpy().astype(int)

    # Create a dictionary to store the results
    labels = ['action_armed_assault', 'action_arrest', 'action_bombing', 'action_infrastructure', 'action_surrender', 'action_seizure', 'action_abduction']
    results = dict(zip(labels, predictions))

    return results

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       inference_target_type
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load the saved model and tokenizer
target_model_path = "target_type/distilBert"
target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
target_model = AutoModelForSequenceClassification.from_pretrained(target_model_path)


target_model.to(device)


def inference_target_type(summary):
    """
    Performs inference on an incident summary to predict target types.

    Args:
        summary: The incident summary text.

    Returns:
        A dictionary with target type labels as keys and their predicted probabilities (0 or 1) as values.
    """

    # Tokenize the input summary
    inputs = target_tokenizer(summary, padding=True, truncation=True, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = target_model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # Get probabilities using sigmoid

    # Convert probabilities to binary predictions (0 or 1) using threshold
    threshold = 0.5
    predictions = (probs > threshold).squeeze().cpu().numpy().astype(int)

    # Create a dictionary to store the results
    labels = ['target_civilians', 'target_maoist', 'target_no_target', 'target_security', 'target_government']
    results = dict(zip(labels, predictions))
    return results


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       predict_counts
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load the tokenizer and model from the saved directory
total_num_tokenizer = T5Tokenizer.from_pretrained('total_injuries-arrests-surrenders-fatalities-abducted/t5small_finetuned_model')
total_num_model = T5ForConditionalGeneration.from_pretrained('total_injuries-arrests-surrenders-fatalities-abducted/t5small_finetuned_model')

total_num_model.to(device)

def extract_number(text):
    match = re.search(r'\b\d+\b', text)
    if match:
        return int(match.group())
    else:
        return 0

def predict_counts(incident_summary):
    questions = [
        ("How many injuries occurred in the incident?", "total_injuries"),
        ("How many arrests were made in the incident?", "total_arrests"),
        ("How many people surrendered in the incident?", "total_surrenders"),
        ("How many fatalities occurred in the incident?", "total_fatalities"),
        ("How many people were abducted in the incident?", "total_abducted")
    ]
    counts = {}
    for question, label in questions:
        input_text = f"question: {question} context: {incident_summary}"
        input_ids = total_num_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        input_ids = input_ids.to(device)
        outputs = total_num_model.generate(input_ids)
        answer = total_num_tokenizer.decode(outputs[0], skip_special_tokens=True)
        count = extract_number(answer)
        counts[label] = count
    return counts

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       predict_damage
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

damage_model_path = 'damage_details_extraction/t5base_finetuned_model'
damage_model = T5ForConditionalGeneration.from_pretrained(damage_model_path)
damage_tokenizer = T5Tokenizer.from_pretrained(damage_model_path)

damage_model.to(device)


def predict_damage(summary):
    # Prepare the input text
    input_text = f"Extract the property damage value from the incident: {summary}"
    input_ids = damage_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids
    input_ids = input_ids.to(device)

    # Generate predictions
    outputs = damage_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)

    # Decode the output
    predicted_damage = damage_tokenizer.decode(outputs[0], skip_special_tokens=True)
    damage_predictions = {
        'value_property_damage': predicted_damage
    }
    return damage_predictions


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       get_location_details
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


location_model_path = 'location_context_extraction/t5base_finetuned_model'
location_model = T5ForConditionalGeneration.from_pretrained(location_model_path)
location_tokenizer = T5Tokenizer.from_pretrained(location_model_path)

location_model.to(device)


# Updated function to get location details including latitude and longitude
def get_location_details(summary):
    """Given a list of location names, constructs a query, calls the Google Geocoding API,
    and returns state, district, subdistrict, town/village, and latitude/longitude of the most specific level."""

    # Prepare the input text
    input_text = f"Extract the location of the incident: {summary}"
    input_ids = location_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids

    input_ids = input_ids.to(device)


    # Generate predictions
    outputs = location_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)

    # Decode the output
    locations = location_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def remove_specific_key_location_words(locations):
        words_to_remove = ["police", "station","dam","river","rivers","forests","forest"]  # Add more words here
        cleaned_locations = locations.lower()
        for word in words_to_remove:
            cleaned_locations = cleaned_locations.replace(word.lower(), "")
        return cleaned_locations

    locations = remove_specific_key_location_words(locations)

    # Google Maps API key
    API_KEY = st.secrets["googlemapsAPI"]
    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


    #query = ', '.join(locations)
    params = {
        'address': locations,
        'key': API_KEY,
        'components': 'country:IN'
    }
    response = requests.get(GEOCODE_URL, params=params)
    if response.status_code != 200:
        print(f"Error in API call: {response.status_code}")
        return None

    data = response.json()
    if data['status'] != 'OK':
        print(f"Geocoding API error: {data['status']}")
        return {
        'Extracted_Locations': locations,
        'state': None,
        'district': None,
        'subdistrict': None,
        'town_village': None,
        'latitude': None,
        'longitude': None,
        'location_Level': "API couldn't find the Extracted_Locations"
    }

    # Initialize components
    state = district = subdistrict = town_village = None
    latitude = longitude = None
    found_level = None  # Keep track of the most specific level found

    # Iterate over results to find the most specific level
    for result in data.get('results', []):
        temp_state = temp_district = temp_subdistrict = temp_town_village = None
        address_components = result['address_components']

        # Map address components
        for component in address_components:
            types = component['types']
            if 'administrative_area_level_1' in types:
                temp_state = component['long_name']
            elif 'administrative_area_level_2' in types:
                temp_district = component['long_name']
            elif 'administrative_area_level_3' in types:
                temp_subdistrict = component['long_name']
            elif 'locality' in types:
                temp_town_village = component['long_name']
            elif 'sublocality' in types and not temp_town_village:
                temp_town_village = component['long_name']

        # Determine the most specific level in this result
        if temp_town_village and found_level not in ['town_village']:
            state = temp_state
            district = temp_district
            subdistrict = temp_subdistrict
            town_village = temp_town_village
            location = result['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            found_level = 'town_village'
        elif temp_subdistrict and found_level not in ['town_village', 'subdistrict']:
            state = temp_state
            district = temp_district
            subdistrict = temp_subdistrict
            town_village = None
            location = result['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            found_level = 'subdistrict'
        elif temp_district and found_level not in ['town_village', 'subdistrict', 'district']:
            state = temp_state
            district = temp_district
            subdistrict = None
            town_village = None
            location = result['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            found_level = 'district'
        elif temp_state and found_level not in ['town_village', 'subdistrict', 'district', 'state']:
            state = temp_state
            district = None
            subdistrict = None
            town_village = None
            location = result['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            found_level = 'state'

        # Break the loop if the most specific level is found
        if found_level == 'town_village':
            break

    return {
        'Extracted_Locations': locations,
        'state': state,
        'district': district,
        'subdistrict': subdistrict,
        'town_village': town_village,
        'latitude': latitude,
        'longitude': longitude,
        'location_Level': found_level,
    }



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       update_dataframe
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------



def update_dataframe(df, return_details, index):
    """Updates the DataFrame with details returned from a function.

    Args:
        df: The pandas DataFrame to update.
        return_details: A dictionary containing the details to add.
        index: The index of the row in the DataFrame to update.
    """
    for column, value in return_details.items():
        df.at[index, column] = value


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       process_dataframe
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def process_dataframe(df, model_inference_function_name, task_description):
    """
    Processes a DataFrame, applying a model inference function to each row and updating the DataFrame.

    Args:
        df: The input DataFrame.
        model_inference_function_name: The name of the function to use for model inference (e.g., get_location_details).  Must be defined in the current scope.
        task_description: A string describing the task being performed.
    """
    start_time = time.time()
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {task_description}"):
        summary = row['Incident_Summary']
        details = model_inference_function_name(summary) # Call the provided function
        if details:
            update_dataframe(df, details, index)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
    print(f"Total time taken for {task_description}: {elapsed_time_str}")
    print(f"{task_description} Completed  and added to the DataFrame.")



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       fetch_data_from_google_sheets
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Function to fetch data from Google Sheets
def fetch_data_from_google_sheets(spreadsheet_name, sheet_name):
    # Set up Google Sheets API credentials
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)

    # Open the spreadsheet and worksheet
    sheet = client.open(spreadsheet_name).worksheet(sheet_name)

    # Fetch all data into a DataFrame
    data = pd.DataFrame(sheet.get_all_records())

    return data
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       predict_damage
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def delete_and_update_sheets(uploaded_df, spreadsheet_name):
    """
    Deletes rows from 'raw_zone_incident_summaries' and updates 'processed_summaries'.

    Args:
        uploaded_df: DataFrame of uploaded incidents.
        spreadsheet_name: Name of the Google Sheet.
    """
    # Set up Google Sheets API credentials
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)

    # Step 1: Save to 'processed_summaries' and ensure no duplicates
    try:
        save_to_google_sheets(uploaded_df, spreadsheet_name, "processed_summaries")
        # print("Uploaded data successfully saved to 'processed_summaries'.")
    except Exception as e:
        print(f"Error while saving to 'processed_summaries': {e}")
        return

    # Step 2: Delete matching rows from 'raw_zone_incident_summaries'
    raw_sheet = client.open(spreadsheet_name).worksheet("raw_zone_incident_summaries")
    raw_data = raw_sheet.get_all_records()
    raw_df = pd.DataFrame(raw_data)

    # Normalize the incident numbers for comparison
    uploaded_incident_numbers = set(uploaded_df['Incident_Number'].astype(str).str.strip())
    raw_df['Incident_Number'] = raw_df['Incident_Number'].astype(str).str.strip()

    # Filter out rows with matching incident numbers
    filtered_raw_df = raw_df[~raw_df['Incident_Number'].isin(uploaded_incident_numbers)]

    # Update the sheet with filtered data (overwrite with remaining rows)
    raw_sheet.clear()  # Clear the existing sheet
    raw_sheet.update([filtered_raw_df.columns.tolist()] + filtered_raw_df.values.tolist())
    print(f"Deleted {len(raw_df) - len(filtered_raw_df)} rows from 'raw_zone_incident_summaries'.")




# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #                                                                       Get data from Google Sheets
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# # Get data from Google Sheets
# spreadsheet_name = "SATP_Data"
# sheet_name = "raw_zone_incident_summaries"
# satp_dat = fetch_data_from_google_sheets(spreadsheet_name, sheet_name)
# satp_data = satp_dat.head(15).copy()


# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #                                                                       get_location_details
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# if satp_data is None:
#     print("No data fetched from Google Sheets.")
#     exit(1)
# else:
#     print("Data fetched from Google Sheets.")

#     process_dataframe(satp_data, infer_perpetrator, task_description="Perpetrator Extraction")
#     process_dataframe(satp_data, inference_action_type, task_description="Action Type Extraction")
#     process_dataframe(satp_data, inference_target_type, task_description="Target Type Extraction")
#     process_dataframe(satp_data, get_location_details, task_description="Location Extraction")
#     process_dataframe(satp_data, predict_counts, task_description="Total Injuries, Arrests, Surrenders, Fatalities, Abducted Extraction")
#     process_dataframe(satp_data, predict_damage, task_description="Damage Extraction")


# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #                                                             save_to_google_sheets main funtion call
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#     save_to_google_sheets(satp_data, "SATP_Data", "ALL Data")
#     delete_and_update_sheets(satp_data, "SATP_Data")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       streamlit
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------



import streamlit as st
import pandas as pd
import time
from streamlit.runtime.scriptrunner import get_script_run_ctx


# Initialize session state for data
if "satp_data" not in st.session_state:
    st.session_state["satp_data"] = None

if "processed" not in st.session_state:
    st.session_state["processed"] = False

# App Title and Subtitle
st.title("Automating Conflict Data Extraction from SATP")
st.subheader("South Asian Terrorism Portal")

device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU/CPU Information
if device == 'cuda':
    st.success("CUDA device detected. Processing will be faster. Proceed with processing larger datasets.")
else:
    st.warning("CPU detected. Processing may take longer. Please choose fewer samples.")

# Scraping Section
st.header("Step 1: Scrape the Data")
st.write("To scrape the raw data, visit [SATP Hosting App](https://satphosting.streamlit.app/).")
if st.button("Confirm Scraping Finished and Saved to Google Sheets"):
    st.success("Data scraping confirmed. Proceed to the next step.")

# Data Input Section
st.header("Step 2: Select Data Source")
data_source = st.radio("Choose your data source:", options=["Google Sheets", "Upload CSV"])

if data_source == "Google Sheets":
    # Google Sheets input
    # Option to process all or specific sample size
    process_all = st.radio("Process all data or a specific sample size?", options=["All", "Specific"])

    if process_all == "Specific":
    # Input for sample size when "Specific" is selected
        sample_size = st.number_input("Enter the number of samples to process:", min_value=1, step=1)
    else:
    # Default to None for processing all data
        sample_size = None

    spreadsheet_name = st.text_input("Spreadsheet Name:", value="SATP_Data")
    sheet_name = st.text_input("Sheet Name:", value="raw_zone_incident_summaries")

    if st.button("Fetch Data"):
        with st.spinner("Fetching data from Google Sheets..."):
            try:
                satp_data = fetch_data_from_google_sheets(spreadsheet_name, sheet_name)
                if sample_size and sample_size <= len(satp_data):
                    satp_data = satp_data.head(sample_size)
                st.session_state["satp_data"] = satp_data
                st.success(f"Data fetched successfully from '{sheet_name}'!")
                st.dataframe(satp_data)
            except Exception as e:
                st.error(f"Error fetching data: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
    if uploaded_file:
        with st.spinner("Loading CSV data..."):
            try:
                satp_data = pd.read_csv(uploaded_file)
                st.session_state["satp_data"] = satp_data
                st.success("CSV data loaded successfully!")
                st.dataframe(satp_data)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

# Password Protection for Data Processing
st.header("Step 3: Process Data")
password = st.text_input("Enter the password to process data:", type="password")
if st.button("Process Data"):
    if password == st.secrets["process_data_password"]:  # Replace with a strong password
        if st.session_state.get("satp_data") is not None:
            with st.spinner("Processing data..."):
                # Run your data processing functions
                process_dataframe(st.session_state["satp_data"], infer_perpetrator, "Perpetrator Extraction")
                process_dataframe(st.session_state["satp_data"], inference_action_type, "Action Type Extraction")
                process_dataframe(st.session_state["satp_data"], inference_target_type, "Target Type Extraction")
                process_dataframe(st.session_state["satp_data"], get_location_details, "Location Extraction")
                process_dataframe(st.session_state["satp_data"], predict_counts, "Counts Extraction")
                process_dataframe(st.session_state["satp_data"], predict_damage, "Damage Extraction")
                st.session_state["processed"] = True
                st.success("Data processing complete!")
        else:
            st.error("No data found. Please load the data first.")
    else:
        st.error("Invalid password. Please try again.")

# Save Processed Data
if st.session_state.get("processed"):
    st.header("Step 4: Save Processed Data")
    save_option = st.radio("Choose your save method:", options=["Google Sheets", "Download as CSV"])
    st.write("### Data to be Saved:")
    st.dataframe(st.session_state["satp_data"])  # Display the DataFrame
        
    if save_option == "Google Sheets":
        if st.button("Save to Google Sheets"):
            with st.spinner("Saving data to Google Sheets..."):
                try:
                    save_to_google_sheets(st.session_state["satp_data"], "SATP_Data", "ALL Data")
                    delete_and_update_sheets(st.session_state["satp_data"], "SATP_Data")
                    st.success("Data saved to Google Sheets successfully!")
                except Exception as e:
                    st.error(f"Error saving data: {e}")
    elif save_option == "Download as CSV":
  # Display the DataFrame
        csv_data = st.session_state["satp_data"].to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed Data", data=csv_data, file_name="processed_data.csv", mime="text/csv")


st.header("Step 5: Dashboard")

dashboard_url = "https://satphosting.streamlit.app/dashboard"  # Replace with your actual dashboard URL

# Create a button with a hyperlink
dashboard_button = f"""
    <a href="{dashboard_url}" target="_blank">
        <button style="background-color: #4CAF50; color: white; border: none; 
                       padding: 10px 20px; text-align: center; 
                       text-decoration: none; font-size: 16px; cursor: pointer; 
                       border-radius: 5px;">
            Go to Dashboard
        </button>
    </a>
"""
st.markdown(dashboard_button, unsafe_allow_html=True)


st.write("---")
st.write("App built by Group 5")
st.write("Mohiddin & Aravind")
