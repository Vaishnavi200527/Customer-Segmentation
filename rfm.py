"""
RFM Feature Engineering Module
Supports PySpark-based distributed aggregation with automatic vectorized Pandas fallback.
Handles log1p transformation, outlier winsorization, and feature standardization.
"""

import sys
import os
import ctypes
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


def _get_short_path(path_str: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(500)
        if ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 500):
            if buf.value and os.path.exists(buf.value):
                return buf.value
    except Exception:
        pass
    return path_str


def initialize_spark_session(driver_memory: str = "4g"):
    """
    Initializes a local PySpark session with explicit loopback binding and worker config.
    """
    safe_py = _get_short_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = safe_py
    os.environ["PYSPARK_DRIVER_PYTHON"] = safe_py
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    try:
        import pyspark
        safe_spark = _get_short_path(os.path.dirname(pyspark.__file__))
        os.environ["SPARK_HOME"] = safe_spark
    except Exception:
        pass
    
    from pyspark.sql import SparkSession
    
    builder = (
        SparkSession.builder
        .appName("CURE_Customer_Intelligence_RFM")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executorEnv.PYSPARK_PYTHON", safe_py)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.python.worker.reuse", "true")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_rfm_with_spark(
    clean_pd: pd.DataFrame,
    spark_session=None,
    progress_callback=None
) -> Tuple[pd.DataFrame, int, str]:
    """
    Aggregates customer-level RFM features using PySpark DataFrame operations.
    """
    from pyspark.sql.functions import (
        col, sum as spark_sum, countDistinct, max as spark_max,
        datediff, lit, to_date
    )
    
    if progress_callback:
        progress_callback(0.15, "Initializing local PySpark session...")
        
    spark = spark_session if spark_session is not None else initialize_spark_session()
    
    if progress_callback:
        progress_callback(0.30, "Transferring clean transactions to Spark DataFrame...")
        
    spark_pd = clean_pd[["Invoice", "InvoiceDate", "Quantity", "Price", "Customer ID"]].copy()
    spark_pd["Invoice"] = spark_pd["Invoice"].astype(str)
    spark_pd["InvoiceDate"] = np.array(pd.to_datetime(spark_pd["InvoiceDate"]).dt.to_pydatetime())
    spark_pd["Quantity"] = spark_pd["Quantity"].astype(float)
    spark_pd["Price"] = spark_pd["Price"].astype(float)
    spark_pd["Customer ID"] = spark_pd["Customer ID"].astype("int64")
    
    sdf = spark.createDataFrame(spark_pd)
    clean_sdf = (
        sdf.filter(
            (~col("Invoice").startswith("C")) &
            (~col("Invoice").startswith("c")) &
            (col("Quantity") > 0) &
            (col("Price") > 0) &
            col("Customer ID").isNotNull() &
            col("InvoiceDate").isNotNull()
        )
        .withColumn("TotalPrice", col("Quantity") * col("Price"))
    )
    
    if progress_callback:
        progress_callback(0.55, "Calculating global reference date in Spark...")
        
    latest_date = clean_sdf.agg(spark_max("InvoiceDate").alias("latest")).first()["latest"]
    if latest_date is None:
        raise ValueError("Could not determine max InvoiceDate from Spark DataFrame.")
        
    analysis_date = pd.Timestamp(latest_date).normalize() + pd.Timedelta(days=1)
    
    if progress_callback:
        progress_callback(0.75, "Performing customer-level groupBy and RFM aggregation...")
        
    rfm_sdf = clean_sdf.groupBy("Customer ID").agg(
        datediff(lit(analysis_date.date()), to_date(spark_max("InvoiceDate"))).alias("Recency"),
        countDistinct("Invoice").alias("Frequency"),
        spark_sum("TotalPrice").alias("Monetary"),
    )
    
    rfm = rfm_sdf.toPandas()
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Customer ID", "Recency", "Frequency", "Monetary"]
    )
    rfm = rfm[
        (rfm["Recency"] >= 0) &
        (rfm["Frequency"] > 0) &
        (rfm["Monetary"] > 0)
    ].copy()
    
    if len(rfm) < 5:
        raise ValueError(f"Fewer than 5 customers remain after RFM aggregation ({len(rfm)}).")
        
    if progress_callback:
        progress_callback(1.0, f"Spark RFM aggregation complete ({len(rfm):,} customers).")
        
    return rfm.reset_index(drop=True), len(clean_pd), str(analysis_date.date())


def build_rfm_with_pandas(
    clean_pd: pd.DataFrame,
    progress_callback=None
) -> Tuple[pd.DataFrame, int, str]:
    """
    Vectorized Pandas aggregation engine with identical logic to PySpark.
    Used when PySpark is unavailable or network sockets are restricted by host policy.
    """
    if progress_callback:
        progress_callback(0.20, "Vectorized aggregation: computing reference date...")
        
    latest_date = clean_pd["InvoiceDate"].max()
    analysis_date = pd.Timestamp(latest_date).normalize() + pd.Timedelta(days=1)
    
    if progress_callback:
        progress_callback(0.50, "Vectorized aggregation: grouping by Customer ID...")
        
    grouped = clean_pd.groupby("Customer ID")
    rfm = grouped.agg(
        Recency=("InvoiceDate", lambda dates: (analysis_date - dates.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()
    
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Customer ID", "Recency", "Frequency", "Monetary"]
    )
    rfm = rfm[
        (rfm["Recency"] >= 0) &
        (rfm["Frequency"] > 0) &
        (rfm["Monetary"] > 0)
    ].copy()
    
    if len(rfm) < 5:
        raise ValueError("Fewer than 5 customers remain after RFM aggregation.")
        
    if progress_callback:
        progress_callback(1.0, f"Vectorized RFM complete ({len(rfm):,} customers).")
        
    return rfm.reset_index(drop=True), len(clean_pd), str(analysis_date.date())


def compute_rfm(
    clean_df: pd.DataFrame,
    engine: str = "spark",
    spark_session=None,
    progress_callback=None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified entry point for RFM computation with automatic fallback.
    """
    used_engine = "PySpark"
    try:
        if engine == "spark":
            rfm, tx_count, ref_date = build_rfm_with_spark(
                clean_df, spark_session=spark_session, progress_callback=progress_callback
            )
        else:
            rfm, tx_count, ref_date = build_rfm_with_pandas(
                clean_df, progress_callback=progress_callback
            )
            used_engine = "Pandas Vectorized"
    except Exception as exc:
        # Graceful fallback to Pandas if Spark cannot initialize (e.g. socket permission or missing Java)
        if engine == "spark":
            if progress_callback:
                progress_callback(0.25, f"Spark unavailable ({str(exc)[:40]}...); switching to high-speed Pandas engine...")
            rfm, tx_count, ref_date = build_rfm_with_pandas(
                clean_df, progress_callback=progress_callback
            )
            used_engine = "Pandas Vectorized (Fallback)"
        else:
            raise exc
            
    metadata = {
        "engine": used_engine,
        "clean_transactions": tx_count,
        "customer_count": len(rfm),
        "reference_date": ref_date,
        "total_revenue": round(float(rfm["Monetary"].sum()), 2),
    }
    return rfm, metadata


def prepare_rfm_features(
    rfm_df: pd.DataFrame,
    winsorize: bool = True,
    winsorize_percentile: float = 0.99,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Transforms RFM features for CURE clustering:
    1. Winsorization (capping extreme 99th percentile outliers in Frequency and Monetary).
    2. Log1p transformation (ln(1 + x)) to address extreme right-skewness.
    3. Standardization (Z-score StandardScaler) to ensure equal scale across R, F, and M.
    """
    df = rfm_df.copy()
    raw_stats = {
        "recency_max": float(df["Recency"].max()),
        "frequency_max": float(df["Frequency"].max()),
        "monetary_max": float(df["Monetary"].max()),
    }
    
    caps = {}
    if winsorize:
        for feature in ["Frequency", "Monetary"]:
            cap_val = float(df[feature].quantile(winsorize_percentile))
            caps[feature] = round(cap_val, 2)
            df[feature] = df[feature].clip(upper=cap_val)
            
    raw_values = df[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)
    
    # Log1p transformation
    log_transformed = np.log1p(raw_values)
    
    # Feature scaling
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X = scaler.fit_transform(log_transformed)
    
    # Numerical safety validation
    if not np.isfinite(X).all():
        raise ValueError("Transformed RFM feature matrix contains NaN or Inf values.")
    if X.shape[1] != 3:
        raise ValueError(f"Expected 3 RFM feature columns, got {X.shape[1]}.")
        
    feature_meta = {
        "winsorized": winsorize,
        "caps": caps,
        "scaler": scaler_type,
        "raw_stats": raw_stats,
        "feature_means": [round(float(m), 4) for m in X.mean(axis=0)],
        "feature_stds": [round(float(s), 4) for s in X.std(axis=0)],
    }
    
    return df, X, feature_meta
