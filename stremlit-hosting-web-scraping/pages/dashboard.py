import json
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import gspread
import geopandas as gpd
from google.oauth2.service_account import Credentials

def main():
    st.set_page_config(
        page_title="Political Incident Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Political Incident Research Dashboard")
    st.markdown("Gain insights into incidents across regions and time with advanced visualizations for political research.")

    # Load the dataset
    @st.cache_data
    # Function to fetch data from Google Sheets
    def fetch_data_from_google_sheets(spreadsheet_name, sheet_name):
        # Set up Google Sheets API credentials
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        service_account_info = st.secrets["google_credentials"]
        creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        client = gspread.authorize(creds)

        # Open the spreadsheet and worksheet
        sheet = client.open(spreadsheet_name).worksheet(sheet_name)

        # Fetch all data into a DataFrame
        data = pd.DataFrame(sheet.get_all_records())
        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        #
        return data

    spreadsheet_name = "SATP_Data"
    sheet_name = "ALL Data"
    data = fetch_data_from_google_sheets(spreadsheet_name, sheet_name)

    # Sidebar for filters
    st.sidebar.header("Filters")
    selected_state = st.sidebar.multiselect(
        "Select State", options=data['state'].unique(), default=data['state'].unique()
    )

    # Handle single year case for the slider
    if data['year'].nunique() > 1:
        selected_year = st.sidebar.slider(
            "Select Year Range",
            int(data['year'].min()),
            int(data['year'].max()),
            (int(data['year'].min()), int(data['year'].max()))
        )
    else:
        only_year = data['year'].iloc[0]
        selected_year = (only_year, only_year)
        st.sidebar.info(f"Data available for only one year: {only_year}")

    # Filter the data
    filtered_data = data[
        (data['state'].isin(selected_state)) & 
        (data['year'].between(*selected_year))
    ]

    # Display filtered data
    st.header("Filtered Data")
    st.dataframe(filtered_data, height=300)

    # Layout for key metrics
    st.header("🔢 Key Metrics")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total Incidents", len(filtered_data))
    col2.metric("Total Fatalities", filtered_data['total_fatalities'].sum())
    col3.metric("Total Injuries", filtered_data['total_injuries'].sum())
    col4.metric("Total Abducted", filtered_data['total_abducted'].sum())
    col5.metric("Total Arrests", filtered_data['total_arrests'].sum())
    col6.metric("Total Surrenders", filtered_data['total_surrenders'].sum())
    col7.metric("Deadliest State", filtered_data.groupby("state")["total_fatalities"].sum().idxmax())

    # # 1. Incident Trend Over Time
    # st.header("📈 Trend of Incidents Over Time")
    # trend_data = filtered_data.groupby(["Date"]).size().reset_index(name="Incident Count")
    # fig_trend = px.line(
    #     trend_data, 
    #     x="Date", y="Incident Count",
    #     title="Daily Incident Trend"
    # )
    # st.plotly_chart(fig_trend)

    # Resample to count incidents by month
    timeline_data = filtered_data.resample('ME', on='Date').size().reset_index(name='Number of Incidents')

    # Create interactive line chart using Plotly
    fig_trend = px.line(
        timeline_data,
        x='Date',
        y='Number of Incidents',
        title='Number of Incidents by Timeline',
        labels={'Date': 'Date', 'Number of Incidents': 'Number of Incidents'},
        markers=True
    )

    # Customize the layout
    fig_trend.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Incidents',
        template='plotly_white',
        title_font=dict(size=20),
        hovermode='x unified'
    )

    # Show the interactive plot
    st.plotly_chart(fig_trend)


    # # 2. Geographic Distribution of Incidents
    # st.header("🗺️ Geographic Distribution of Incidents")
    # state_counts = filtered_data.groupby("state").size().reset_index(name="Incident Count")
    # geojson_file_path = r"india-states.json"  # Replace with actual GeoJSON path
    # try:
    #     with open(geojson_file_path, "r") as file:
    #         india_geojson = json.load(file)
    #     geojson_states = [feature['properties']['ST_NM'] for feature in india_geojson['features']]
    #     state_counts = state_counts[state_counts['state'].isin(geojson_states)]
    #     fig_geo = px.choropleth_mapbox(
    #         state_counts,
    #         geojson=india_geojson,
    #         locations="state",
    #         featureidkey="properties.ST_NM",
    #         color="Incident Count",
    #         title="State-wise Incident Distribution",
    #         color_continuous_scale="Viridis",
    #         mapbox_style="carto-positron",
    #         center={"lat": 22.9734, "lon": 78.6569},
    #         zoom=4.5,
    #         opacity=0.6,
    #     )
    #     st.plotly_chart(fig_geo)
    # except FileNotFoundError:
    #     st.error("GeoJSON file not found. Please update the file path.")



    # Load data and aggregate by state (assuming `data` is already defined)
    state_data = filtered_data.groupby('state', as_index=False).agg({
        'Incident_Number': 'count',
        'total_injuries': 'sum',
        'total_arrests': 'sum',
        'total_surrenders': 'sum',
        'total_fatalities': 'sum',
        'total_abducted': 'sum'
    })
    state_data.rename(columns={'Incident_Number': 'total_incidents'}, inplace=True)
    
    # Load the India states GeoJSON as a GeoDataFrame
    gdf = gpd.read_file("https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson")
    
    # Standardize the names in both dataframes
    gdf['NAME_1'] = gdf['NAME_1'].str.title().str.strip()
    state_data['state'] = state_data['state'].str.title().str.strip()
    
    # Merge the data on st_nm and state
    # Merge all states from gdf with state_data, even if they don't match
    merged_gdf = gdf.merge(state_data, left_on='NAME_1', right_on='state', how='left')
    
    # Fill missing values with 0 to ensure we have valid numeric data for plotting
    merged_gdf[['total_incidents', 'total_injuries', 'total_arrests', 'total_surrenders',
                'total_fatalities', 'total_abducted']] = merged_gdf[[
                    'total_incidents', 'total_injuries', 'total_arrests', 'total_surrenders',
                    'total_fatalities', 'total_abducted'
                ]].fillna(0)
    geojson_dict = json.loads(merged_gdf.to_json())
    
    merged_gdf.rename(columns={'NAME_1': 'State'}, inplace=True)
    
    fig_geo = px.choropleth_mapbox(
        merged_gdf,
        geojson=geojson_dict,
        locations='State',               # use the column with state names
        featureidkey='properties.NAME_1', # match this to the GeoJSON property
        color='total_incidents',          # column to color by
        color_continuous_scale=["white", "red"],
        mapbox_style="carto-positron",
        zoom=3.5,
        center={"lat": 20, "lon": 78},
        opacity=0.7,
        hover_data={
            'State': True,
            'total_incidents': True,
            'total_injuries': True,
            'total_arrests': True,
            'total_surrenders': True,
            'total_fatalities': True,
            'total_abducted': True
        }
    )
    
    fig_geo.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_geo)
    
    

    # 3. Actions and Perpetrators
    st.header("🎯 Actions by Perpetrators")
    action_columns = [
        "action_armed_assault", "action_bombing", 
        "action_infrastructure", "action_surrender"
    ]
    action_counts = filtered_data[action_columns].sum()
    fig_actions = px.bar(
        action_counts, 
        x=action_counts.index, y=action_counts.values, 
        title="Distribution of Actions by Type",
        labels={"x": "Action Type", "y": "Count"}
    )
    st.plotly_chart(fig_actions)

    # # 4. Word Cloud of Incident Summaries
    # st.header("☁️ Word Cloud of Incident Summaries")
    # all_text = " ".join(filtered_data['Incident_Summary'].dropna())
    # wordcloud = WordCloud(
    #     background_color='white', width=800, height=400
    # ).generate(all_text)
    # fig_wc, ax = plt.subplots(figsize=(12, 6))
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.axis("off")
    # st.pyplot(fig_wc)

    # 5. Proportions of Incident Types
    st.header("📊 Proportions of Incident Types")
    incident_type_columns = ["action_armed_assault", "action_bombing", "action_infrastructure", "action_surrender"]
    action_counts = filtered_data[incident_type_columns].sum()
    fig_pie = px.pie(
        action_counts, 
        values=action_counts.values, 
        names=action_counts.index,
        title="Incident Type Distribution"
     )
    st.plotly_chart(fig_pie)   

    # 6. Heatmap of Incidents by State and Year
    st.header("🔥 Heatmap of Incidents by State and Year")
    heatmap_data = filtered_data.groupby(["state", "year"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 7. Fatalities and Injuries Over Time
    st.header("📊 Fatalities and Injuries Over Time")
    fatality_data = filtered_data.groupby("Date")[["total_fatalities", "total_injuries"]].sum().reset_index()
    fig_fatalities = px.bar(
        fatality_data, 
        x="Date", y=["total_fatalities", "total_injuries"], 
        title="Daily Fatalities and Injuries",
        labels={"value": "Count", "Date": "Date"}, barmode="group"
    )
    st.plotly_chart(fig_fatalities)

    # Insights on most affected states and peak times
    st.header("📊 State Rankings and Monthly Distribution")
    col5, col6 = st.columns(2)
    state_rankings = filtered_data.groupby("state").size().sort_values(ascending=False).head(10)
    col5.bar_chart(state_rankings)

    monthly_trends = filtered_data.groupby("month").size()
    col6.line_chart(monthly_trends)
    
    # Download Button for Filtered Data
    st.sidebar.header("📥 Download Filtered Data")
    if st.sidebar.button("Download CSV"):
        filtered_data.to_csv("filtered_data.csv", index=False)
        st.sidebar.success("Filtered data saved as 'filtered_data.csv'.")

if __name__ == "__main__":
    main()
