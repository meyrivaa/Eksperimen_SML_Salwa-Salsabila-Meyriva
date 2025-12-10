import pandas as pd
import numpy as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
  df = df.drop_duplicates()
  df = df.fillna(method="ffill")
  
  df_encoded = df.copy()
  label_encoders = {}
  
  cat_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
  
  for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
  
  scaler = StandardScaler()
  num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
  num_cols.remove("churned")
  
  df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

  return df_encoded

def save_processed(df, out_path):
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    raw_path = "dataset_raw/ecommerce-customer-churn_dataset.csv"
    out_path = "preprocessing/dataset_preprocessing/ecommerce-customer-churn_dataset_preprocessing.csv"

    df = load_data(raw_path)
    processed = preprocess_data(df)
    save_processed(processed, out_path)

    print("Preprocessing selesai! File disimpan di:", out_path)
