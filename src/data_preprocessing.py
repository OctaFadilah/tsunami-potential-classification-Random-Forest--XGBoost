import numpy as np
import pandas as pd
from global_land_mask import globe
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path_gempa, path_tsunami):
    # Load raw datasets
    datagempa = pd.read_csv(path_gempa, delimiter=';')
    katalogtsunami = pd.read_csv(path_tsunami, delimiter=';')

    # Filter earthquake data based on minimum magnitude and maximum depth
    datagempa2 = datagempa[(datagempa['MT_Magnitude(Mw)'] >= 5.3) & (datagempa['MT_Depth(km)'] <= 68)]
    datagempa3 = datagempa2[['Origin_Time(UT)', 'MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)','Latitude(deg)','Rake','Strike','Dip']].copy()

    # Format the origin time in the tsunami catalog
    katalogtsunami['Origin_Time(UT)'] = (katalogtsunami['Year'].astype(str) + '/' +
                                  katalogtsunami['Mo'].astype(str).str.zfill(2) + '/' +
                                  katalogtsunami['Dy'].astype(str).str.zfill(2) + ',' +
                                  katalogtsunami['Hr'].astype(str).str.zfill(2) + ':' +
                                  katalogtsunami['Mn'].astype(str).str.zfill(2) + ':' +
                                  katalogtsunami['Sec'].astype(str).replace(',', '.', regex=True))

    catalogtsunami2 = katalogtsunami[['Origin_Time(UT)', 'Latitude', 'Longitude']].copy()

    datagempa3['Origin_Time(UT)'] = pd.to_datetime(datagempa3['Origin_Time(UT)'].str.replace('"', ''), format='%Y/%m/%d,%H:%M:%S.%f')
    catalogtsunami2['Origin_Time(UT)'] = pd.to_datetime(catalogtsunami2['Origin_Time(UT)'].str.replace('"', ''), format='%Y/%m/%d,%H:%M:%S.%f')

    datagempa3['Time_dt'] = datagempa3['Origin_Time(UT)']
    catalogtsunami2['Time_dt'] = catalogtsunami2['Origin_Time(UT)']
    catalogtsunami2['match_found'] = 1

    # Sort data by time for merging
    datagempa3 = datagempa3.sort_values('Time_dt')
    catalogtsunami2 = catalogtsunami2.sort_values('Time_dt')

    # Merge data using merge_asof to establish tsunami labels
    datagempa4 = pd.merge_asof(
        datagempa3,
        catalogtsunami2[['Time_dt', 'match_found']],
        on='Time_dt',
        direction='nearest',
        tolerance=pd.Timedelta('1.5min')
    )

    # Create binary label (1 if matched with a tsunami event, 0 otherwise)
    datagempa4['Label_Tsunami'] = datagempa4['match_found'].fillna(0).astype(int)
    datagempa4 = datagempa4.drop(columns=['match_found', 'Time_dt'])

    # Filter events occurring offshore using global_land_mask
    def is_it_sea(lat, lon):
        return 0 if globe.is_land(lat, lon) else 1

    datagempa4['di_laut'] = datagempa4.apply(lambda x: is_it_sea(x['Latitude(deg)'], x['Longitude(deg)']), axis=1)
    datagempa5 = datagempa4[datagempa4['di_laut'] == 1]

    # Filter geographical bounding box for Japan
    min_lat, max_lat = 24.0, 46.0
    min_lon, max_lon = 122.0, 148.0

    datagempa6 = datagempa5[
        (datagempa5['Latitude(deg)'] >= min_lat) & (datagempa5['Latitude(deg)'] <= max_lat) &
        (datagempa5['Longitude(deg)'] >= min_lon) & (datagempa5['Longitude(deg)'] <= max_lon)
    ]

    # Drop specific spatial outlier
    datagempa7 = datagempa6.drop(index=[1029], errors='ignore')

    # Function to classify fault types based on the Rake angle
    def klasifikasi_sesar_3kelas(rake_val):
        if pd.isna(rake_val):
            return np.nan
        if isinstance(rake_val, str):
            try:
                rake = float(rake_val.split(';')[0])
            except ValueError:
                return np.nan
        else:
            rake = float(rake_val)

        if 45 <= rake <= 135:
            return 1  # Reverse Fault
        elif -135 <= rake <= -45:
            return 0  # Normal Fault
        else:
            return 2  # Strike-Slip Fault

    datagempa7['Fault_Type_New'] = datagempa7['Rake'].apply(klasifikasi_sesar_3kelas)
    if 'Fault_Type_Encoded' in datagempa7.columns:
        datagempa7 = datagempa7.drop('Fault_Type_Encoded', axis=1)

    # One-Hot Encoding for fault types
    datagempa7 = pd.get_dummies(datagempa7, columns=['Fault_Type_New'], prefix='Fault', dtype=int)
    return datagempa7

def prepare_datasets(df):
    # Dataset 2: Includes basic parameters + fault types
    X = df[['MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)', 'Latitude(deg)', 'Fault_0', 'Fault_1', 'Fault_2']]
    y = df['Label_Tsunami']
    
    # Dataset 1: Baseline with basic seismic parameters only
    A = df[['MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)','Latitude(deg)']]
    B = df['Label_Tsunami']

    # Train-test split (80:20 ratio with stratification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42, stratify=B)
    
    return X_train, X_test, y_train, y_test, A_train, A_test, B_train, B_test
