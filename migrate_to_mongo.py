import pandas as pd
import glob
import os
import pymongo
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("No MONGO_URI found in environment variables")

def load_data_from_csv():
    """
    Loads data from the 'data/' directory.
    """
    print("Loading raw CSV data...")
    base_path = "data"
    
    # 1. Enrolment Data
    enrolment_files = glob.glob(os.path.join(base_path, "api_data_aadhar_enrolment", "**", "*.csv"), recursive=True)
    print(f"Found {len(enrolment_files)} enrolment files.")
    df_enrolment = pd.concat((pd.read_csv(f) for f in enrolment_files), ignore_index=True)
    
    # 2. Demographic Data
    demographic_files = glob.glob(os.path.join(base_path, "api_data_aadhar_demographic", "**", "*.csv"), recursive=True)
    print(f"Found {len(demographic_files)} demographic files.")
    df_demographic = pd.concat((pd.read_csv(f) for f in demographic_files), ignore_index=True)
    
    # 3. Biometric Data
    biometric_files = glob.glob(os.path.join(base_path, "api_data_aadhar_biometric", "**", "*.csv"), recursive=True)
    print(f"Found {len(biometric_files)} biometric files.")
    df_biometric = pd.concat((pd.read_csv(f) for f in biometric_files), ignore_index=True)
    
    return df_enrolment, df_demographic, df_biometric

VALID_STATES = {
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
}

def clean_state_names(df):
    df['state'] = df['state'].astype(str).str.strip().str.title()
    df['state'] = df['state'].str.replace(r'\s+', ' ', regex=True)
    
    typo_map = {
        'West Bangal': 'West Bengal', 'West Bengli': 'West Bengal', 'Westbengal': 'West Bengal',
        'West  Bengal': 'West Bengal', 'Uttaranchal': 'Uttarakhand', 'Odisha': 'Odisha', 
        'Orissa': 'Odisha', 'Pondicherry': 'Puducherry', 'Tamilnadu': 'Tamil Nadu',
        'Andhra Pradesh': 'Andhra Pradesh', 'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu', 'Jammu And Kashmir': 'Jammu and Kashmir',
        'Jammu & Kashmir': 'Jammu and Kashmir', 'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu', 'Raja Annamalai Puram': 'Tamil Nadu',
        'Madanapalle': 'Andhra Pradesh', 'Puttanahalli': 'Karnataka', 'Nagpur': 'Maharashtra',
    }
    df['state'] = df['state'].replace(typo_map)
    df = df[df['state'].isin(VALID_STATES)]
    return df

def preprocess_data(df_enrolment, df_demographic, df_biometric):
    print("Preprocessing and Aggregating data (preserving age columns)...")
    
    def parse_dates(df):
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df['date'] = df['date'].apply(lambda x: x.replace(day=1) if pd.notnull(x) else x)
        return df

    # --- Enrolment Processing ---
    df_enrolment = parse_dates(df_enrolment)
    df_enrolment['Total_Enrolment'] = (df_enrolment['age_0_5'] + df_enrolment['age_5_17'] + df_enrolment['age_18_greater'])
    # UPGRADE: Keep specific age groups
    enrol_cols = ['Total_Enrolment', 'age_0_5', 'age_5_17', 'age_18_greater']
    enrol_agg = df_enrolment.groupby(['date', 'state', 'district'])[enrol_cols].sum().reset_index()

    # --- Demographic Processing ---
    df_demographic = parse_dates(df_demographic)
    df_demographic['Total_Demographic'] = (df_demographic['demo_age_5_17'] + df_demographic['demo_age_17_'])
    # UPGRADE: Keep specific age groups
    demo_cols = ['Total_Demographic', 'demo_age_5_17', 'demo_age_17_']
    demo_agg = df_demographic.groupby(['date', 'state', 'district'])[demo_cols].sum().reset_index()

    # --- Biometric Processing ---
    df_biometric = parse_dates(df_biometric)
    df_biometric['Total_Biometric'] = (df_biometric['bio_age_5_17'] + df_biometric['bio_age_17_'])
    # UPGRADE: Keep specific age groups
    bio_cols = ['Total_Biometric', 'bio_age_5_17', 'bio_age_17_']
    bio_agg = df_biometric.groupby(['date', 'state', 'district'])[bio_cols].sum().reset_index()

    # --- Merge All ---
    merged = pd.merge(enrol_agg, demo_agg, on=['date', 'state', 'district'], how='outer')
    merged = pd.merge(merged, bio_agg, on=['date', 'state', 'district'], how='outer')
    
    merged.fillna(0, inplace=True)
    merged = clean_state_names(merged)

    # Cleaning logic
    def is_valid_name(x):
        return isinstance(x, str) and not x.replace('.', '', 1).isdigit()
    merged['district'] = merged['district'].astype(str).str.strip().str.title()
    merged = merged[merged['state'].apply(is_valid_name)]
    merged = merged[merged['district'].apply(is_valid_name)]
    
    # Re-aggregate all columns including the detailed age ones
    numeric_cols = [
        'Total_Enrolment', 'age_0_5', 'age_5_17', 'age_18_greater',
        'Total_Demographic', 'demo_age_5_17', 'demo_age_17_',
        'Total_Biometric', 'bio_age_5_17', 'bio_age_17_'
    ]
    merged = merged.groupby(['date', 'state', 'district'])[numeric_cols].sum().reset_index()
    
    return merged

def migrate():
    try:
        raw_enrol, raw_demo, raw_bio = load_data_from_csv()
        final_df = preprocess_data(raw_enrol, raw_demo, raw_bio)
        print(f"Data processed. Rows: {len(final_df)}")
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    try:
        print(f"Connecting to MongoDB: {MONGO_URI}")
        client = pymongo.MongoClient(MONGO_URI)
        db = client["uidai_db"]
        collection = db["district_metrics"]
        
        print("Clearing existing collection...")
        collection.delete_many({})
        
        print("Inserting records...")
        records = final_df.to_dict("records")
        if records:
            collection.insert_many(records)
            print(f"Successfully inserted {len(records)} records with aggregated age columns.")
            
            # Indexes
            collection.create_index([("date", pymongo.ASCENDING)])
            collection.create_index([("state", pymongo.ASCENDING)])
            collection.create_index([("district", pymongo.ASCENDING)])
        else:
            print("No records to insert.")
            
    except Exception as e:
        print(f"MongoDB Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    migrate()
