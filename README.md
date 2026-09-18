# Indian Road Accident Severity Dashboard

## Problem Statement

Road accident data in India is a mess to work with. The raw numbers are out there, but they are usually stuck
in Excel sheets or government PDFs with no easy way to visualize them. You end up spending hours just trying
to figure out which states have the most accidents or whether rain actually makes things worse.

This app fixes that. Drag in your CSV and within a few seconds you can see breakdowns by state, severity,
road type, time of day, and more. No coding needed once it is running.

---

## How to Install and Run

**Step 1 -- Make sure you have Python installed (3.9 or above works fine)**

`ash
python --version
`

**Step 2 -- Clone or download this folder to your machine**

`ash
git clone <your-repo-url>
cd indian_road_accident_severity
`

**Step 3 -- Create a virtual environment (optional but recommended)**

`ash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
`

**Step 4 -- Install all the dependencies**

`ash
pip install -r requirements.txt
`

**Step 5 -- Run the app**

`ash
streamlit run app.py
`

It should open in your browser at http://localhost:8501 automatically.

---

## How to Use the App

When the app opens, you will see a sidebar on the left. Here is what each page does:

- **Home and Overview** -- Landing page. Shows four KPI cards at the top and two charts below. Good starting point.
- **Upload and Explore** -- Go here if you have your own CSV or Excel file. Click Browse files, pick your file,
  and the app shows a column breakdown, data types, and missing value counts.
- **Visualizations** -- Pick a chart from the dropdown. Look at accidents by road type, weather, vehicle type,
  time of day, and more. Charts are interactive so you can hover over them to see exact numbers.
- **Filter and Analyze** -- Filter by state, severity, road type, and speed limit. The table and chart update
  in real time. You can also download the filtered data as a CSV.
- **About** -- Basic info about the project and the tools used.

The sidebar also has global settings to toggle raw data tables, change the chart color theme, and adjust how
many top states to show.

---

## Diagrams

These diagrams should be added to this section once drawn. Placeholder descriptions are below.

### System Architecture Diagram

Shows how the different parts of the app connect. At the top is the user browser running Streamlit.
Below that is the Python backend that handles data loading, processing, and chart generation.
The data either comes from the uploaded file or from the built-in sample data generator.

`
[ User Browser ]
      |
      v
[ Streamlit Frontend (UI components, widgets) ]
      |
      v
[ Python Backend ]
   |         |
   v         v
[ Uploaded  [ Sample Data
   File ]     Generator  ]
   |         |
   v         v
[ Pandas DataFrame ]
      |
      v
[ Plotly / Matplotlib Charts ]
      |
      v
[ Rendered back to browser ]
`

### Data Flow Diagram

Shows the path your data takes from the moment you upload a file to when you see a chart on screen.

`
[ CSV or Excel File ]
        |
        v
[ File Uploader (Streamlit widget) ]
        |
        v
[ pandas read_csv or read_excel ]
        |
        v
[ DataFrame validation and null check ]
        |
        v
[ Filter logic (state, severity, road type, speed) ]
        |
        v
[ Aggregation using groupby ]
        |
        v
[ Plotly chart or Streamlit table ]
        |
        v
[ Displayed in the browser ]
`

### User Workflow Diagram

Shows what a typical user does when they open the app.

`
Open app
   |
   v
Land on Home page --> See KPI summary
   |
   v
Go to Upload and Explore --> Upload CSV file
   |
   v
Go to Visualizations --> Pick chart type --> Look at patterns
   |
   v
Go to Filter and Analyze --> Apply filters --> Find specific records
   |
   v
Download filtered CSV
`

---

## Important Tables

### Accident Records Table

This is the main table -- either your uploaded data or the generated sample.

| Column | What it means |
|---|---|
| Accident_ID | A unique number for each accident record |
| State | Which Indian state the accident happened in |
| Road_Type | National highway, state highway, city road, or rural road |
| Weather | Weather conditions at the time -- clear, rainy, foggy, or cloudy |
| Time_of_Day | Morning, afternoon, evening, or night |
| Vehicle_Type | What kind of vehicle was involved |
| Junction_Type | Whether the accident was at a T-junction, crossroads, or open stretch |
| Speed_Limit_kmph | The posted speed limit on that road in km/h |
| Persons_Involved | Total number of people in the accident |
| Severity | How bad the accident was -- Minor, Serious, or Fatal |
| Fatalities | Number of people who died |
| Injuries | Number of people injured |
| Year | The year the accident happened |

### Filtered Summary Table (Filter and Analyze page)

Appears after you apply filters. Groups results by state.

| Column | What it means |
|---|---|
| State | The state name |
| Total_Accidents | How many accidents matched your filters in that state |
| Fatalities | Total deaths in the filtered records for that state |
| Injuries | Total injuries in the filtered records for that state |

### Column Info Table (Upload and Explore page)

Shows what is in your uploaded file.

| Column | What it means |
|---|---|
| Column | The column name from your file |
| Dtype | The data type -- int, float, object, etc. |
| Non-Null | How many rows have actual values (not blank) |

---

## All the Pipes and Connections

Here is how data moves through the app from input to output.

**Inputs:**
- A CSV or Excel file you upload via the file uploader
- Widget selections such as multiselect dropdowns, sliders, radio buttons, and checkboxes
- If no file is uploaded the generate_sample_data function creates a fake dataset with 500 rows

**What happens in the middle:**
1. The file gets read by pandas read_csv or read_excel
2. On the Filter page your widget values get applied as a boolean mask on the dataframe
3. The filtered dataframe gets aggregated using groupby and agg to create summary tables
4. The dataframe gets passed to Plotly functions that turn it into charts

**Outputs:**
- Plotly charts rendered inside st.plotly_chart
- Pandas dataframes shown inside st.dataframe
- A downloadable CSV file via st.download_button on the Filter and Analyze page
- KPI numbers shown in st.metric cards on the Home page

The sidebar settings like show_raw_data, color_theme, and top_n_states act as global switches
that affect how things look across all pages.

---

## Files in This Folder

| File | What it does |
|---|---|
| app.py | The main application file. Run this with Streamlit to start the app. |
| requirements.txt | Lists every Python library the app needs. Use with pip install -r requirements.txt. |
| README.md | This file. Explains how to set up and use the project. |

If you have your own dataset drop it in this folder and upload it through the app interface.

---

## Notes and Troubleshooting

**Module not found error when running the app**
You probably missed a package. Run pip install -r requirements.txt again and check that your
virtual environment is activated.

**The app opens but shows a blank page**
Try refreshing the browser. If that does not work stop the terminal process with Ctrl+C
and run streamlit run app.py again.

**My uploaded file is not loading**
Make sure it is a .csv or .xlsx file. Files with blank rows at the top or weird encoding sometimes
cause issues. Open it in Excel, save it again as CSV (UTF-8), and try re-uploading.

**Charts look weird or do not show up**
Try switching the chart color theme in the sidebar. Some themes work better depending on your browser.

**The app is running slow**
If you uploaded a very large file (50,000+ rows) some operations will take a few seconds.
For very large datasets consider sampling the data before uploading.

**Port already in use error**
Run this to use a different port:
`ash
streamlit run app.py --server.port 8502
`
