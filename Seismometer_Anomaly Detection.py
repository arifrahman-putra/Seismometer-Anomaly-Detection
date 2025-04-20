"""
This script was made as an implementation the conference paper:
"A Design of Seismometer Anomaly Detection System Based on Frequency-domain Features"
Published in IEEE Xplore, 16 April 2025.
DOI:  10.1109/ISRITI64779.2024.10963616

For more details, refer to the published paper.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from obspy.clients.filesystem.sds import Client
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import plot_trigger, trigger_onset
from obspy import read_inventory
from obspy.core.stream import Stream
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from obspy import UTCDateTime
from obspy.core import read
import pickle

# Seismic folder access
client = Client("C:\\Users\\Seismic Folder") # your seismic mseed folder

# Seismic Metadata initialization
Stations = ["ABC", "DEF", "GHI"] # seismic station codes
Channels = ["SHE", "SHN", "SHZ"] # seismometer channels
StartDate = UTCDateTime("2025-04-10T21:00:00.000000Z") # starting date
EndDate = UTCDateTime("2025-04-11T21:00:00.000000Z") # ending date
Network = "YZ" # your seismic observation network


# 1-SVM pickle model initialization
One_SVM_path = "model_1SVM.pkl"  # Update path as needed

with open(One_SVM_path, 'rb') as f:
    One_SVM_model = pickle.load(f)

def predict_with_One_SVM(model, data):
    # Predict using the 1-SVM model
    return model.predict(data)

# Recursive STA-LTA initialization
def label_traces(trace, sta_length, lta_length, trigger_threshold):
    df = trace.stats.sampling_rate
    cft = recursive_sta_lta(trace.data, int(sta_length * df), int(lta_length * df))
    triggers = trigger_onset(cft, trigger_threshold, trigger_threshold * 0.5)

    if len(triggers) > 0:
        label = -1  # earthquake-contaminated recording label

    else:
        label = 1  # earthquake-free recording label

    return label

P_Stations=[] # Processed station codes
P_Channels=[] # Processed seismometer channels
P_StartTimes=[] # Processed recording starting times
P_EndTimes=[] # Processed recording ending times
EQ_labels=[] # earthquake detection labels
avg_NLNM_Dev_A=[] # Averaged NLNM Deviations (4-8s)
avg_NLNM_Dev_B=[] # Averaged NLNM Deviations (18-22s)
avg_NLNM_Dev_C=[] # Averaged NLNM Deviations (90-110s)
avg_NLNM_Dev_D=[] # Averaged NLNM Deviations (200-500s)
avg_NHNM_Dev_A=[] # Averaged NHNM Deviations (4-8s)
avg_NHNM_Dev_B=[] # Averaged NHNM Deviations (18-22s)
avg_NHNM_Dev_C=[] # Averaged NHNM Deviations (90-110s)
avg_NHNM_Dev_D=[] # Averaged NHNM Deviations (200-500s)
Detection_labels=[] # anomaly detection labels
Descriptions=[] # data descriptions

# obtain NLNM values at four significant period bands
df_NLNM=pd.read_excel("NLNM_values.xlsx")
NLNM_per=df_NLNM["Period"].tolist() # NLNM periods
NLNM_val=df_NLNM["NLNM"].tolist() # NLNM values

# obtain NHNM values at four significant period bands
df_NHNM=pd.read_excel("NHNM_values.xlsx")
NHNM_per=df_NHNM["Period"].tolist() # NHNM periods
NHNM_val=df_NHNM["NHNM"].tolist() # NHNM values


for StationID in Stations:
    for Channel in Channels:
        Hours = int((EndDate-StartDate)/3600)
        for Hour in range(Hours):
            StartTime = StartDate + (Hour*3600) # recording starting time
            EndTime = StartTime + 3600 # recording ending time
            st = client.get_waveforms(Network, StationID, "*", Channel, StartTime, EndTime)
            tr = st[0]
            link_inv = "C:\\Inventory\\" + StationID + ".xml" # your xml inventories folder
            inv = read_inventory(link_inv)
            ppsd = PPSD(tr.stats, metadata=inv)
            ppsd.add(st)

            EQ_label = label_traces(tr, 1.0, 10.0, 2.0)

            P_Stations.append(StationID)
            P_Channels.append(Channel)
            P_StartTimes.append(StartTime)
            P_EndTimes.append(EndTime)
            EQ_labels.append(EQ_label)

            if EQ_label == -1: # If earthquake is detected
                avg_NLNM_Dev_A.append(None)
                avg_NLNM_Dev_B.append(None)
                avg_NLNM_Dev_C.append(None)
                avg_NLNM_Dev_D.append(None)
                avg_NHNM_Dev_A.append(None)
                avg_NHNM_Dev_B.append(None)
                avg_NHNM_Dev_C.append(None)
                avg_NHNM_Dev_D.append(None)
                Detection_labels.append(0) # label earthquake contaminated data with 0
                Descriptions.append("Earthquake occurrence")


            else: # If earthquake is not detected:
                NLNM_Dev_A = []  # 4-8 s (secondary microseism)
                for k in range(4, 9, 1):
                    for j in range(len(NLNM_per)):
                        if NLNM_per[j] == k:
                            NLNM_Dev_A.append(ppsd.extract_psd_values(k)[0][0] - NLNM_val[j])
                avg_A_L = sum(NLNM_Dev_A) / len(NLNM_Dev_A)
                avg_NLNM_Dev_A.append(avg_A_L)

                NLNM_Dev_B = []  # 18-22 s (primary microseism)
                for k in range(18, 23, 1):
                    for j in range(len(NLNM_per)):
                        if NLNM_per[j] == k:
                            NLNM_Dev_B.append(ppsd.extract_psd_values(k)[0][0] - NLNM_val[j])
                avg_B_L = sum(NLNM_Dev_B) / len(NLNM_Dev_B)
                avg_NLNM_Dev_B.append(avg_B_L)

                NLNM_Dev_C = []  # 90-110 s (surface waves)
                for k in range(90, 111, 2):
                    for j in range(len(NLNM_per)):
                        if NLNM_per[j] == k:
                            NLNM_Dev_C.append(ppsd.extract_psd_values(k)[0][0] - NLNM_val[j])
                avg_C_L = sum(NLNM_Dev_C) / len(NLNM_Dev_C)
                avg_NLNM_Dev_C.append(avg_C_L)

                NLNM_Dev_D = []  # 200-500 s (normal modes)
                for k in range(200, 501, 25):
                    for j in range(len(NLNM_per)):
                        if NLNM_per[j] == k:
                            NLNM_Dev_D.append(ppsd.extract_psd_values(k)[0][0] - NLNM_val[j])
                avg_D_L = sum(NLNM_Dev_D) / len(NLNM_Dev_D)
                avg_NLNM_Dev_D.append(avg_D_L)


                NHNM_Dev_A = []  # 4-8 s (secondary microseism)
                for k in range(4, 9, 1):
                    for j in range(len(NHNM_per)):
                        if NHNM_per[j] == k:
                            NHNM_Dev_A.append(ppsd.extract_psd_values(k)[0][0] - NHNM_val[j])
                avg_A_H = sum(NHNM_Dev_A) / len(NHNM_Dev_A)
                avg_NHNM_Dev_A.append(avg_A_H)

                NHNM_Dev_B = []  # 18-22 s (primary microseism)
                for k in range(18, 23, 1):
                    for j in range(len(NHNM_per)):
                        if NHNM_per[j] == k:
                            NHNM_Dev_B.append(ppsd.extract_psd_values(k)[0][0] - NHNM_val[j])
                avg_B_H = sum(NHNM_Dev_B) / len(NHNM_Dev_B)
                avg_NHNM_Dev_B.append(avg_B_H)

                NHNM_Dev_C = []  # 90-110 s (surface waves)
                for k in range(90, 111, 2):
                    for j in range(len(NHNM_per)):
                        if NHNM_per[j] == k:
                            NHNM_Dev_C.append(ppsd.extract_psd_values(k)[0][0] - NHNM_val[j])
                avg_C_H = sum(NHNM_Dev_C) / len(NHNM_Dev_C)
                avg_NHNM_Dev_C.append(avg_C_H)

                NHNM_Dev_D = []  # 200-500 s (normal modes)
                for k in range(200, 501, 25):
                    for j in range(len(NHNM_per)):
                        if NHNM_per[j] == k:
                            NHNM_Dev_D.append(ppsd.extract_psd_values(k)[0][0] - NHNM_val[j])
                avg_D_H = sum(NHNM_Dev_D) / len(NHNM_Dev_D)
                avg_NHNM_Dev_D.append(avg_D_H)

                features_dict = {"NLNM_Dev_A": [avg_A_L], "NLNM_Dev_B": [avg_B_L],
                                 "NLNM_Dev_C": [avg_C_L], "NLNM_Dev_D": [avg_D_L],
                                 "NHNM_Dev_A": [avg_A_H], "NHNM_Dev_B": [avg_B_H],
                                 "NHNM_Dev_C": [avg_C_H], "NHNM_Dev_D": [avg_D_H]}

                df_features = pd.DataFrame(features_dict)
                pred_label=predict_with_One_SVM(One_SVM_model, df_features)
                Detection_labels.append(pred_label)

                if pred_label == 1:
                    Descriptions.append("Normal data")

                else:
                    Descriptions.append("Anomalous data")


dataset_dict={"StationID": P_Stations, "Channel": P_Channels,"StartTime": P_StartTimes, "EndTime": P_EndTimes,
              "NLNM_Dev_A": avg_NLNM_Dev_A, "NLNM_Dev_B": avg_NLNM_Dev_B, "NLNM_Dev_C": avg_NLNM_Dev_C, "NLNM_Dev_D": avg_NLNM_Dev_D,
              "NHNM_Dev_A": avg_NHNM_Dev_A, "NHNM_Dev_B": avg_NHNM_Dev_B, "NHNM_Dev_C": avg_NHNM_Dev_C, "NHNM_Dev_D": avg_NHNM_Dev_D,
              "EQ_label": EQ_labels, "Detection_label": Detection_labels, "Description": Descriptions}

df_result = pd.DataFrame(dataset_dict)

out_path = "AnomalyDetection_Result.xlxs"
df_result.to_excel(out_path) # save result to xlsx file in your desired directory