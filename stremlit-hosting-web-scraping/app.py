import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Initialize session state for storing scraped data and save state
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "total_incidents" not in st.session_state:
    st.session_state.total_incidents = 0
if "save_initiated" not in st.session_state:
    st.session_state.save_initiated = False

# Scraping function
def scrape_satp_data(base_url, years, months):
    data = []
    month_to_number = {
        "Jan": "01", "Feb": "02", "Mar": "03",
        "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09",
        "Oct": "10", "Nov": "11", "Dec": "12"
    }
    for year in years:
        for month in months:
            url = f"{base_url}-{month}-{year}"
            with st.spinner(f"Scraping: {url}"):
                response = requests.get(url)
                if response.status_code != 200:
                    st.warning(f"Failed to fetch data for {month} {year}: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                coverpage_news = soup.find_all('div', class_='more')
                coverpage_date = soup.find_all('td', style="width: 15%;")

                if len(coverpage_news) != len(coverpage_date):
                    st.warning(f"Mismatch in incidents ({len(coverpage_news)}) and dates ({len(coverpage_date)}) for {month} {year}.")
                    continue

                incidents_by_date = {}
                for date, incident in zip(coverpage_date, coverpage_news):
                    incident_summary = incident.get_text().strip()
                    incident_summary = re.sub(r"\s+", " ", incident_summary).replace("Read less...", "")

                    raw_date = date.get_text().strip()
                    day = raw_date.split('-')[-1].strip()
                    month_number = month_to_number[month]
                     # month_number = f"{months.index(month) + 1:02}"
                    formatted_date = f"{year}-{month_number}-{day.zfill(2)}"

                    if formatted_date not in incidents_by_date:
                        incidents_by_date[formatted_date] = 0
                    incidents_by_date[formatted_date] += 1

                    nn = f"{incidents_by_date[formatted_date]:02}"
                    incident_number = f"I{month_number}{day.zfill(2)}{year[-2:]}{nn}"

                    data.append({
                        "Incident_Number": incident_number,
                        "Date": formatted_date,
                        "Incident_Summary": incident_summary
                    })

    return pd.DataFrame(data), len(data)

# # Save to Google Sheets function
# def save_to_google_sheets(data, spreadsheet_name, sheet_name):
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
#     service_account_info = st.secrets["google_credentials"]
#     creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
#     client = gspread.authorize(creds)

#     try:
#         sheet = client.open(spreadsheet_name).worksheet(sheet_name)
#     except gspread.exceptions.WorksheetNotFound:
#         spreadsheet = client.open(spreadsheet_name)
#         sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=data.shape[1])
#         sheet.append_row(list(data.columns))

#     existing_data = pd.DataFrame(sheet.get_all_records())
#     new_rows = data if existing_data.empty else data[~data['Incident_Number'].isin(existing_data['Incident_Number'])]
    
#     if not new_rows.empty:
#         sheet.append_rows(new_rows.values.tolist())
#         return f"Uploaded {len(new_rows)} new rows to Google Sheet."
#     else:
#         return "No new incidents found to upload."

# save_to_google_sheets
def save_to_google_sheets(data, spreadsheet_name, sheet_name):
    # Set up Google Sheets API credentials
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    service_account_info = st.secrets["google_credentials"]
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
    if existing_data.empty and 'Incident_Number' in existing_data.columns:
        # If no data exists in the sheet, upload the entire dataset
        sheet.clear()
        new_rows = data
    elif existing_data.empty and 'Incident_Number' not in existing_data.columns:
        sheet.clear()
        new_rows = data
        sheet.append_row(list(data.columns))
    else:
        if 'Incident_Number' in existing_data.columns and 'Incident_Number' in data.columns:
            # Find new rows based on incident number
            existing_incident_numbers = set(existing_data['Incident_Number'])
            new_rows = data[~data['Incident_Number'].isin(existing_incident_numbers)]
        else:
            new_rows = None

    if not new_rows.empty:
        # Prepare new data for batch update
        batch_data = new_rows.values.tolist()  # Convert DataFrame to a list of lists
        # Insert new rows
        sheet.append_rows(batch_data)
        return f"Uploaded {len(new_rows)} new rows to Google Sheet {sheet_name}."
    else:
        return f"No new incidents found to upload for the sheet {sheet_name}."


# Streamlit app
st.title("Scrape Incident Summaries from South Asian Terrorism Portal")

# Base URL dropdown
base_url = st.selectbox(
    "Select Base URL",
    ["https://www.satp.org/terrorist-activity/india-maoistinsurgency"],
    index=0
)

# Year and Month multiselect
years = st.multiselect("Select Years", [str(year) for year in range(2017, 2025)], default=["2017", "2018"])
months = st.multiselect("Select Months", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], default=["Jan", "Feb"])

# Scrape Data button
if st.button("Scrape Data"):
    if not years or not months:
        st.error("Please select at least one year and one month.")
    else:
        with st.spinner("Scraping data..."):
            scraped_data, total_incidents = scrape_satp_data(base_url, years, months)
            st.session_state.scraped_data = scraped_data
            st.session_state.total_incidents = total_incidents
        st.success(f"Total Incidents Scraped: {st.session_state.total_incidents}")
        if not st.session_state.scraped_data.empty:
            st.dataframe(st.session_state.scraped_data)
            csv = st.session_state.scraped_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='scraped_data.csv',
                mime='text/csv',
            )

# Save to Google Sheets functionality
if st.session_state.scraped_data is not None:
    if st.button("Initiate Save to Google Sheets"):
        st.session_state.save_initiated = True
        
if st.session_state.save_initiated:
    st.write("### Data to be Saved:")
    st.dataframe(st.session_state.scraped_data)  # Display the DataFrame
    csv = st.session_state.scraped_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='scraped_data.csv',
        mime='text/csv',
    )
    
    with st.form("save_to_sheets_form"):
        st.write("Save the scraped data to Google Sheets (requires password).")
        password = st.text_input("Enter Password:", type="password")
        save_button = st.form_submit_button("Save to Google Sheets")

        if save_button:
            if password == st.secrets["scrape_password"]:
                with st.spinner("Saving to Google Sheets..."):
                    result = save_to_google_sheets(st.session_state.scraped_data, "SATP_Data", "raw_zone_incident_summaries")
                st.success(result)
                st.session_state.save_initiated = False  # Reset after successful save
            else:
                st.error("Incorrect password!")
