# 🚀 Team Setup & Execution Guide
## Project: CURE Customer Intelligence Platform (Big Data Analytics)

This guide provides step-by-step instructions for any team member to set up, run, and explore the **CURE Customer Intelligence Platform** on their local machine.

---

## 📋 System Prerequisites

Before running the project, ensure you have:
1. **Python 3.10 or 3.11 (64-bit)** installed on your machine.
   - [Download Python](https://www.python.org/downloads/)
   - *Important during installation:* Make sure to check the box **"Add Python to PATH"**.
2. **Operating System:** Windows 10/11, macOS, or Linux.
3. **Java (Optional for PySpark):** OpenJDK 11 or 17.
   - *Note:* The project includes an **automatic high-speed Pandas fallback**, so if Java is not installed, the application will automatically run using the vectorized Pandas engine without failing.

---

## ⚡ Quick 5-Step Execution Guide

### Step 1: Open PowerShell or Terminal in the Project Directory
Navigate to the root folder of the project:
```powershell
cd "c:\Users\VAISHNAVI KUMBHAR\Downloads\CURE_Customer_Segmentation_Corrected"
```
*(Or your team member's local path where the project was extracted).*

---

### Step 2: Create and Activate a Virtual Environment

Creating an isolated virtual environment ensures clean dependencies:

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> **Tip:** If PowerShell displays an execution policy error like `running scripts is disabled on this system`, run this command first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> Then run `.\.venv\Scripts\Activate.ps1` again.

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Packages

Upgrade pip and install the required dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Run the Complete 9-Suite Verification Test

Before launching the web interface, run the automated test suite to ensure all algorithms and data pipelines pass:
```powershell
python test_pipeline.py
```

You should see:
```text
[TEST 1] Verifying module imports and syntax...           --> PASS
[TEST 2] Testing transaction cleaning and row audit...    --> PASS
[TEST 3] Testing RFM aggregation & log1p scaling...       --> PASS
[TEST 4] Testing CURE representative point selection...   --> PASS
[TEST 5] Testing full CURE clustering execution...        --> PASS
[TEST 6] Testing clustering quality evaluation...         --> PASS
[TEST 7] Testing statistical profiling & personas...      --> PASS
[TEST 8] Testing Plotly chart generation...               --> PASS
[TEST 9] Testing data exports and report generation...    --> PASS

ALL 9 TEST SUITES COMPLETED AND PASSED WITH ZERO ERRORS!
```

---

### Step 5: Launch the Streamlit Web Application

Start the web application:
```powershell
python -m streamlit run app.py
```

Within a few seconds, Streamlit will display:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open your browser (Chrome, Edge, Firefox, Brave) and navigate to:
```
http://localhost:8501
```

---

## 🖥️ How to Use the Dashboard (Team Walkthrough)

Once the application opens in your browser:

### 1. Ingest Data & Build the RFM Matrix
- In the left sidebar, keep **`Packaged UCI Dataset (Local)`** selected.
- Click the blue button **"🚀 Load Dataset & Compute RFM"** in **Step 1**.
- The app will clean transaction logs, remove cancellations, filter positive quantities/prices, and calculate customer-level **Recency**, **Frequency**, and **Monetary** values.

### 2. Run the CURE Clustering Algorithm
- In the sidebar, you can inspect the **4 CURE Parameters**:
  - **Target Clusters ($k$):** `4` (Default)
  - **Sample Size ($s$):** `1000` (Default)
  - **Representatives per Cluster ($c$):** `8` (Default)
  - **Shrink Factor ($\alpha$):** `0.30` (Default)
- Click the button **"🔮 Execute CURE Clustering"** in **Step 2**.
- In ~5 seconds, the progress bar will complete and unlock the dashboard.

### 3. Explore the 5 Analytics Tabs
1. **📊 Executive Overview:**
   - Review 6 KPI cards (Total Customers: 5,878, Clean Transactions: 805,549, Revenue: £17.74M, Silhouette: 0.2632).
   - Examine the **Customer Volume Distribution Bar Chart** and **Share Donut Chart**.
   - Read the 4 actionable customer persona cards:
     - **Cluster 1:** Loyal Core Customers (24.1%)
     - **Cluster 2:** Hibernating / Low Engagement (71.2%)
     - **Cluster 3:** Champions & VIP Spenders (3.1%)
     - **Cluster 4:** New & Promising Shoppers (1.7%)
2. **📈 RFM Analytics:**
   - Inspect the 3 boxplots for Recency, Frequency, and Monetary spreads.
   - Review the detailed **Segment Statistical Profiles** table.
3. **🌐 3D Cluster Space:**
   - Click, drag, and rotate the interactive **3D Scatter Plot** of the customer clusters in standardized log-RFM space.
4. **🔎 Customer Explorer:**
   - Search for individual customers by Customer ID (e.g. `15194` or `17703`).
   - Filter by specific clusters, order frequency, or spend threshold.
5. **📥 Export Data:**
   - One-click downloads for **Customer Segments (CSV)** and **Cluster Summary (CSV)**.

---

## ❓ Frequently Asked Questions & Troubleshooting

### Q1: What does `WARN Shell: Did not find winutils.exe` in the terminal mean?
* **Answer:** This is a standard, **completely harmless** informational notice printed by Apache Hadoop when running PySpark on Windows without native C binaries. PySpark automatically uses standard built-in Java classes without any issues.

### Q2: How do I stop or restart the application?
* **To stop:** In your terminal/PowerShell window, press `Ctrl + C` (type `Y` if prompted to terminate batch job).
* **To restart:** Run `python -m streamlit run app.py`.

### Q3: How do I share exported customer files with my team?
* Navigate to the **📥 Export Data** tab in the dashboard and click:
  - **`📥 Customer Segments (CSV)`** for the complete 5,878 customer list with assigned cluster personas.
  - **`📥 Cluster Summary (CSV)`** for statistical median Recency, Frequency, and Monetary spend metrics.

### Q4: Where is the complete project report for our professor?
* Open [`Mini_Project_Report.md`](Mini_Project_Report.md) in the project root directory. It contains the complete 6-chapter university report, answers to the 4 core problem questions, and a literature review of 10 research papers.
