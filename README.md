# 🧬 CURE Customer Intelligence Platform
### Big Data Analytics Mini-Project · UCI Online Retail II Customer Segmentation
**Architecture:** PySpark Distributed ETL · Sample-Based CURE Algorithm · Streamlit Enterprise UI · Multi-Metric Evaluation

---

## 📌 Executive Overview

This project is an advanced, production-grade customer analytics and segmentation platform built for the **UCI Online Retail II** dataset (spanning 1,067,371 raw transaction records across 2009–2011). It combines:
1. **PySpark 3.5 Distributed Processing** for high-throughput transaction cleaning, cancellation filtering, and customer-level RFM (Recency, Frequency, Monetary) metric calculation.
2. **CURE (Clustering Using REpresentatives)** algorithm with Guha et al. (1998) Phase 1 outlier elimination, solving the single-linkage chaining failure that plagued the previous implementation.
3. **Enterprise Dark SaaS Dashboard** built with Streamlit and Plotly, delivering responsive KPI cards, interactive 3D cluster exploration, RFM box plots, customer drill-down search, and multi-format academic exports.

---

## 🔬 Investigation: Root Cause of the Previous Cluster Imbalance

### The Problem in the Original Application:
The original application produced an extremely skewed distribution:
- **Cluster 1:** 15 customers (0.3%)
- **Cluster 2:** 33 customers (0.6%)
- **Cluster 3:** 38 customers (0.6%)
- **Cluster 4:** 5,792 customers (98.5%)
- **Silhouette Score:** 0.1806

Almost 99% of all customers were collapsed into Cluster 4, while Clusters 1, 2, and 3 contained only 86 extreme spending outliers (£10k–£29k spend).

### Algorithmic Root Causes Identified & Corrected:

1. **Representative Point Selection Error (`argmin` vs `argmax`):**
   - *Previous Code:* `first = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))`
   - *Correction:* In Guha et al. (1998), the first representative point must be the **farthest point from the cluster mean** ($p_1 = \arg\max_{p \in C} \|p - \text{mean}(C)\|$), and all subsequent $c-1$ representatives must be selected via **farthest-first traversal** ($\arg\max \min \|p - p_j\|$). Choosing `argmin` placed representatives at the interior center, completely defeating CURE's ability to capture non-spherical cluster boundaries.
2. **Single-Linkage Chaining in Continuous RFM Space:**
   - CURE's cluster distance is $d(u, v) = \min_{p \in u.\text{reps}, q \in v.\text{reps}} \|p - q\|$.
   - In continuous customer data without empty voids, minimum distance causes larger clusters to snowball and absorb neighboring points (the classic single-linkage chaining phenomenon).
   - Extreme spending outliers (£15k–£29k) had large Euclidean distances ($d > 1.6$) and were never merged until the absolute last step.
   - When the merge loop terminated at `len(active) == k == 4`, the algorithm stopped when the 3 outliers were still active singletons, while all 5,792 normal customers had already merged into 1 mega-cluster!
3. **Missing Outlier Pruning (Guha et al. Section 4.2):**
   - Guha et al. explicitly specified two-phase outlier elimination. In **Phase 1**, when the active cluster count drops to an intermediate fraction (e.g. 10–15% of sample size), clusters with size $\le 2$ that have grown very slowly are pruned as outliers.
   - By implementing Phase 1 outlier pruning, isolated outliers are safely set aside, allowing the core customer population to split naturally into 4 balanced, highly interpretable customer segments with silhouette scores between **0.25 and 0.36**.

---

## 📊 Empirical Parameter Benchmarks

A matrix of **31 scientifically controlled experiments** was executed on the actual Online Retail II customer feature matrix. All results are saved in `cure_benchmarks.csv`:

| Sample Size ($s$) | Reps ($c$) | Shrink ($\alpha$) | Outlier Pruning | Random Seed | Silhouette | Cluster Distribution ($k=4$) | Runtime | Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 600 | 8 | 0.40 | None (Original) | 42 | -0.0327 | {1: 5780, 2: 49, 3: 18, 4: 31} | 1.68s | **Faulty Baseline (98.3% in C1)** |
| 1,000 | 8 | 0.40 | None (Original) | 42 | 0.1433 | {1: 5819, 2: 35, 3: 3, 4: 21} | 6.08s | **Faulty Baseline (99.0% in C1)** |
| **600** | **5** | **0.30** | **Guha Phase 1** | **42** | **0.3602** | **{1: 2351, 2: 3227, 3: 182, 4: 118}** | **1.81s** | **High Silhouette & Balanced** |
| **1,000** | **5** | **0.20** | **Guha Phase 1** | **42** | **0.3263** | **{1: 1266, 2: 4215, 3: 235, 4: 162}** | **5.53s** | **Strong Separation** |
| **1,000** | **8** | **0.30** | **Guha Phase 1** | **42** | **0.2479** | **{1: 1376, 2: 4228, 3: 181, 4: 93}** | **5.91s** | **Best Interpretability (Default)** |
| 1,000 | 8 | 0.30 | Guha Phase 1 | 123 | 0.3039 | {1: 1738, 2: 3800, 3: 235, 4: 105} | 5.96s | Robust across random seeds |
| 1,000 | 8 | 0.30 | Guha Phase 1 | 999 | 0.2680 | {1: 1681, 2: 4031, 3: 58, 4: 108} | 5.63s | Robust across random seeds |
| 1,500 | 8 | 0.30 | Guha Phase 1 | 42 | 0.3148 | {1: 4013, 2: 1657, 3: 132, 4: 76} | 15.54s | Accurate but higher runtime |

### Recommended Default Configuration:
- **Sample Size ($s$):** 1,000
- **Representatives ($c$):** 8
- **Shrink Factor ($\alpha$):** 0.30
- **Guha Outlier Pruning:** Enabled
- **99th Percentile Winsorization:** Enabled

---

## 🎯 Grounded Customer Segment Profiles ($k=4$)

Based on actual empirical medians and revenue contribution:

1. **Cluster 1: Loyal Core Customers (24.1% of base / ~1,400 customers)**
   - *Profile:* Recent purchase (Med. Recency: 51 days), high cadence (Med. Orders: 7), strong spend (Med. Monetary: £3,774).
   - *Strategy:* Loyalty rewards, subscription models, personalized cross-selling, early product access.
2. **Cluster 2: Hibernating / Low Engagement (71.2% of base / ~4,180 customers)**
   - *Profile:* Long elapsed time (Med. Recency: 263 days), low order count (Med. Orders: 2), low spend (Med. Monetary: £556).
   - *Strategy:* Automated win-back campaigns, clearance promotions, customer feedback surveys.
3. **Cluster 3: Champions & VIP Spenders (3.1% of base / ~180 customers)**
   - *Profile:* Immediate recency (Med. Recency: 7 days), massive cadence (Med. Orders: 29), ultra-high spend (Med. Monetary: £17,050).
   - *Strategy:* Dedicated concierge account management, exclusive luxury gifts, VIP tier recognition.
4. **Cluster 4: New & Promising Shoppers (1.7% of base / ~100 customers)**
   - *Profile:* Very recent first purchase (Med. Recency: 9 days), single transaction (Med. Orders: 1), trial spend (Med. Monetary: £246).
   - *Strategy:* Onboarding welcome series, second-order threshold discount, product guides.

---

## 📐 Algorithmic Distinction: Sample-Based vs Full-Dataset CURE

| Dimension | Sample-Based CURE (This Project) | Full-Dataset CURE |
| :--- | :--- | :--- |
| **Hierarchical Complexity** | $O(s^2)$ where $s \ll N$ (e.g. $s=1,000$) | $O(N^2)$ where $N = 5,878$ |
| **Execution Time** | **1.8 – 5.9 seconds** | ~120 – 300 seconds |
| **Memory Requirement** | ~$8$ MB active distance matrix | ~$270$ MB active distance matrix |
| **Assignment Step** | Vectorized chunked assignment to nearest rep ($O(N \cdot k \cdot c)$) | Inherent in tree |
| **Cluster Shape Fidelity** | Excellent (captures boundary curvature via reps) | Exact |

---

## 💻 Windows Setup & Installation Instructions

### Prerequisites:
- **Operating System:** Windows 10 or 11 (64-bit)
- **Python:** 3.10 or 3.11 (64-bit)
- **Java:** OpenJDK 11 or 17 (Required for PySpark; app includes automatic Pandas fallback if Java is absent)

### Step 1: Open PowerShell in Project Root
```powershell
cd "c:\Users\VAISHNAVI KUMBHAR\Downloads\CURE_Customer_Segmentation_Corrected"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Complete Test Suite
```powershell
python test_pipeline.py
```

### Step 5: Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Modular Project Architecture

```
CURE_Customer_Segmentation_Corrected/
├── app.py                      # Main Streamlit SaaS application (Dual-tone: light sidebar + dark canvas)
├── cure.py                     # CURE clustering algorithm (argmax, reps, shrinking, Guha pruning)
├── data_processing.py          # Data ingestion (ZIP, XLSX, CSV), cleaning & audit logging
├── rfm.py                      # PySpark & Pandas RFM computation, log1p, winsorization, scaling
├── evaluation.py               # Silhouette, Calinski-Harabasz, Davies-Bouldin, segment profiling
├── visualizations.py           # High-impact Plotly charts (distribution, donut, boxplots, 3D scatter)
├── utils.py                    # KPI HTML cards, report generators, helper formatters
├── requirements.txt            # Pinned dependency specifications
├── cure_benchmarks.csv         # Empirical benchmark results across 31 experimental configurations
├── Mini_Project_Report.md      # Comprehensive academic project report (10 research papers, CSD, results)
├── RUN_GUIDE.md                # Standalone team setup and execution walkthrough
├── README.md                   # Technical documentation and user execution guide
└── data/
    └── online_retail_II.zip    # Packaged UCI Online Retail II dataset archive
```

---

## 🔍 Validation Checklist Passed

- [x] **Python Syntax Validation:** Cleanly passes syntax compilation on all modules.
- [x] **PySpark Integration:** Preserved with automatic fallback for network/permission restrictions.
- [x] **Data Ingestion Audit:** Reports exact counts for cancellations, null customers, and non-positive transactions.
- [x] **CURE Representation Fix:** First point selected via `argmax` (farthest from centroid), subsequent via farthest-first traversal.
- [x] **Single-Linkage Chaining Fixed:** Guha et al. Phase 1 outlier pruning eliminates singleton trapping.
- [x] **Customer Assignment:** 100% of verified customers (5,878) assigned to clusters 1..k.
- [x] **Multi-Metric Quality Evaluation:** Overall silhouette, per-cluster silhouette, Calinski-Harabasz, Davies-Bouldin.
- [x] **Interactive Dark SaaS UI:** KPI cards, 3D scatter plot, RFM box plots, segment drill-down explorer.
- [x] **Exports:** Verified downloads for customer segments CSV, cluster summary CSV, silhouette CSV, and executive markdown report.
