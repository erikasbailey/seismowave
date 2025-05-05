# Import libraries

import pathlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import read
import datetime
import glob
from scipy import stats
import pickle

from sklearn.preprocessing import MinMaxScaler
import random
import time

# Configuration Parameters

#General paramters (from Minio et al.'s config file) - Sea data not used in the final analysis as it crops out part of the ROI

# down-limit of latitude for Sicilian Channel Sea  
lat1 = float(35.10)

# up-limit of latitude for Sicilian Channel Sea     
lat2 = float(38.09) 

# down-limit of longitude for Sicilian Channel Sea  
lon1 = float(11.45)

# up-limit of longitude  for Sicilian Channel Sea    
lon2 = float(15.87) 

# directory containing hindcast map (netCDF4)    
local_sea = "Input/Sea data/Data"

# Onset of the period to analyse
starttime = UTCDateTime('2018-01-01 00:00:00')

# End of the period to analyse   
endtime = UTCDateTime('2018-04-30 23:59:59') 

# name of the output file for cleaned rms and sea data     !!!!!!!!!!!!!!!! CHANGE NAMES FOR BATCHES - 4 month intervals  
file_out = "all_data_batch1.pickle"

# name of the output file for errors model  !!!!!!!!!!!!!!!! CHANGE NAMES FOR BATCHES - 4 month intervals
file_out_err = "all_data_err_batch1.pickle"

# step in hours for the period to analyze      
nhours = float(24) 

# format of the sea files
fileformat_sea = "{time}_*.nc"

# date format in the name of the sea files   
dateformat = "%Y%m%d"  

#list of network codes - seismic data
network = ['IV'] 

#list of station codes - seismic data     
stations = ['CAVT', 'MMGO', 'CLTA', 'HPAC', 'HAGA', 'SOLUN', 'CSLB', 'MUCR', 'MSRU', 'AIO', 'PZIN', 'MPNC', 'MSDA', 'WDD']   

#list of calibration values - seismic data
sensitivity = [1572860000, 299640000, 1179650000, 1500000000, 600000000, 471860000, 377486000, 1500000000, 600000000, 1500000000, 1500000000, 1500000000, 480400000, 621678000]   

#list of channel codes - seismic data 
channels = ['HHZ', 'HHN', 'HHE']

# directory containing seismic traces (mseed))    
local_seism = "Input/Seismic data/Data"

# format of the seismic files  
file_format_rms = "{station}..{channel}.D.{time}.mseed"

#time window in seconds
time_window= 81.92 

# list of the limits of the frequency bands   
freq = [(0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.95), (0.95, 1.1), (1.1, 1.25), (1.25, 1.4), (1.4, 1.55), (1.55, 1.7), (1.7, 1.85), (1.85, 2.0)] 

# step in hour used for downsampling the rms time series  
rms_step = 1

# threshold to delete ambigous rms values  
rms_thr = 1e-9 

# maximum number of nan for each column of RMS dataframe  
row_thr = 5000

# skewness limit used to apply box-cox transformation    
skew_limit = 0.7 

# locaton to save pre-processed data
folder_save = "Processed Data"


# dataframe of earthquakes for past period    
eqCatalog= "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"

# number of hours used to clean the dataset from earthquake influencev   
hours_del = float(2)

# Magnitude over-threshold for regional eartquakes   
MagMed = float(5.5) 

# Magnitude over-threshold for global eartquakes  
MagWorld = float(7)

# down-limit of latitude for Mediterreanean Sea   
mLat = float(29)

# up-limit of latitude for Mediterreanean Sea 
MLat = float(47)

# down-limit of longitude for Mediterreanean Sea   
mLon = float(5)

# updown-limit of longitude for Mediterreanean Sea    
MLon = float(37)


# Initialise some variables that are used repeatedly in different functions - not in config file

#Define the temporal period to analyze
time_period=np.arange(starttime, endtime, 3600*nhours) # 3600*24 means reading one file per day
time_period

# Define the sampling rate to speed up computation
sampling_rate = 100

def LocalSeaAnalysis():
	# LocalSeaAnalysis Equivalent function in Original Code.
	# Combines the SWH data from the .nc files for the required time period, and within the region of interest wrt latitude and longitude.
	# There is no need to pass all the parameters in the original code to the function, as they are defined in the cells above.

	# Created on: 01/11/2024 using parts of the code by Minio et al., modifications to it, and new additions.

    #Initialize an empty dictionary to storage all sea data through rows
    df_sea = {'Y': None, 'time': [], 'lat':[], 'lon':[]}

    #For each date/time
    for time_day in time_period:
        #Convert time_day to string to facilitate looking up the correct file
        time_day_str = time_day.strftime(dateformat)
        #Get the time since it is in the file name
        time_str = time_day.strftime('%H%M')
        # print(f"Converted time_day to string: {time_day_str}")

        file_pattern = f"{local_sea}/{time_str}_{time_day_str}.nc"
        #print(f"Looking for files matching pattern: {file_pattern}")
              
        #Retrieve file
        fl = glob.glob(file_pattern)
        #print(f"Retrieved files: {fl}")

        #Check if files were found
        if not fl:
            #print(f"No files found for {time_day_str}, skipping...")
            continue

        #Read sea height map
        for file in fl:
            # print(f"Loading dataset from file: {file}")
            #Open the file using netCDF4 library
            f = Dataset(file)
    
            #Extract height data
            height = f.variables['VHM0'][:] #slice all rows from VHM0 (SWH) Data
            height[height.mask] = np.nan #mask invalid height values
            # print(f"Extracted height data, shape: {height.shape}")
    
            #Convert time variable to datetime
            time = pd.to_datetime(f.variables['time'][:], unit='s', utc=True)
            # print(f"Converted time variable to datetime: {time}")
    
            #Create latitude and longitude arrays - array of np.shape(height)[0] - first row of height - evenly spaced latitude values between valid_min (start) and valid_max (stop)
            latitude = np.linspace(f.variables['latitude'].valid_min, f.variables['latitude'].valid_max, np.shape(height)[0])
            longitude = np.linspace(f.variables['longitude'].valid_min, f.variables['longitude'].valid_max, np.shape(height)[1])
            # print(f"Created latitude and longitude arrays: lat shape {latitude.shape}, lon shape {longitude.shape}")
    
            #Create latitude/longitude grids - ij for matrix indexing of output
            Lat, Lon = np.meshgrid(latitude, longitude, indexing='ij')
            # print(f"Generated latitude/longitude grids: Lat shape {Lat.shape}, Lon shape {Lon.shape}")
    
            #Store data
            if df_sea['Y'] is None:
                df_sea['Y'] = height[np.newaxis, ...]  # Create a new axis to make it 3D
                df_sea['time'].append(time)  # Append the current time
                # print("Initialized df_sea['Y'] with current height data.")
                # print(f"After first storage: df_sea['Y'].shape = {df_sea['Y'].shape}, height.shape = {height.shape}")
            else:
                # print(f"Before concatenation: df_sea['Y'].shape = {df_sea['Y'].shape}, height.shape = {height.shape}")
                df_sea['Y'] = np.concatenate((df_sea['Y'], height[np.newaxis, ...]), axis=0)  # Concatenate new height data along the first axis
                df_sea['time'].append(time)  # Append the current time
                # print("Appended new height data to df_sea['Y'].") 

    
            #Close the dataset
            f.close()
            #print(f"Closed dataset for file: {file}")

    #Extract Latitude and Longitude Indices
    lat_indices = np.argwhere((latitude >= lat1) & (latitude <= lat2)).flatten()
    lon_indices = np.argwhere((longitude >= lon1) & (longitude <= lon2)).flatten()
    #print(f"Latitude indices: {lat_indices}, Longitude indices: {lon_indices}")
    #print(f"length of lat_indices, lon_indices: {len(lat_indices), len(lon_indices)}")

    #Filter Data Based on Indices
    mr, Mr = np.min(lat_indices), np.max(lat_indices)
    mc, Mc = np.min(lon_indices), np.max(lon_indices)
    #print(f"Filtering data with indices: lat {mr}-{Mr}, lon {mc}-{Mc}")

    df_sea['Y'] = df_sea['Y'][:, mr:Mr + 1, mc:Mc + 1]  # Include upper bounds
    df_sea['lat'] = Lat[mr:Mr + 1, mc:Mc + 1]
    df_sea['lon'] = Lon[mr:Mr + 1, mc:Mc + 1]
    #print("Data filtered based on specified latitude and longitude ranges.")

    #Reformat Data
    df_sea['Y'] = np.array([np.flipud(m) for m in df_sea['Y']]) # adjust orientation
    df_sea['lat'] = np.flipud(df_sea['lat'])
    df_sea['lon'] = np.flipud(df_sea['lon'])
    #print("Data matrices rotated and flipped to adjust orientation.")

    #Retrieve Shape of Sea Wave Matrix
    org_shape = df_sea['Y'].shape
    #print(f"Original shape of the sea wave matrix: {org_shape}")

    #Success message
    print("OK! Sea wave time series were extracted.")
    
    #Return Results
    print("Returning results...")
    return df_sea, org_shape
    
def LocalRMSAnalysis():
    #Initialise an empty dataframe to store all RMS data
    df_seism = None
    print("Initialized empty DataFrame for storing RMS data.")

    for time_day in time_period:
        
        day_start_time = time.time()
        
        #Convert time_day to string to facilitate looking up the correct file
        time_day_str = time_day.strftime(dateformat)
        print(f"Processing data for day: {time_day_str}")
        
        window_samples = int(time_window * sampling_rate) #Calculate number of samples in a window
        step_samples = int(rms_step * sampling_rate) #Calculate the number of samples between each rms step
        
        data_col = None

        #Iterate over each channel
        for ch in channels:
            
            #Iterate over each station and sensitivity
            for staz, calib in zip(stations, sensitivity):
                
                #retrieve the file path
                file_path = glob.glob(f"{local_seism}/{file_format_rms.format(station=staz, channel=ch, time=time_day_str)}", recursive=True)
                if not file_path:
                    print(f"No file found for station {staz}, channel {ch} on {time_day_str}")
                    continue
                file_path = file_path[0]
                
                #Read and merge the seismic traces - if there are multiple traces, these are merged into one. interpolate missing values
                st = read(file_path)
                st.merge(method=0, fill_value='interpolate')
                
                #Calculate RMS for each frequency band
                for f1, f2 in freq:
                    #print(f"Processing frequency band: {f1}-{f2} Hz")

                    freq_band_start_time = time.time()
                    
                    #Header for the dataframe
                    data_header = pd.DataFrame([[staz, ch[-1], f"{round(f1, 2)}-{round(f2, 2)}"]], 
                                               columns=['station', 'component', 'freq'])

                    #Preprocess, filter, and normalize the trace
                    tr = st.copy().slice(starttime=time_day, endtime=time_day + 3600 * nhours)
                    tr.detrend('demean').detrend('linear')
                    tr.filter('bandpass', freqmin=f1, freqmax=f2, corners=4)
                    tr = tr[0]
                    tr.data = tr.data / calib

                    rms_start_time = time.time()
                    
                    #RMS calculation using numpy
                    n_samples = len(tr.data)

                    #Compute RMS in chunks, including partial windows
                    rms = []
                    times_rms = []
                    for i in range(0, n_samples, step_samples):
                        #Check if we are at the end of the signal (partial window)
                        if i + window_samples <= n_samples:
                            window = tr.data[i:i + window_samples] #Full window
                        else:
                            #Partial window: pad with zeros
                            window = np.pad(tr.data[i:], (0, window_samples - len(tr.data[i:])), mode='constant')

                        rms.append(np.sqrt(np.mean(window ** 2)))
                        times_rms.append(tr.stats.starttime + (i + window_samples / 2) / sampling_rate) #caculate corresponding timestamp (midpoint of window)
                    
                    rms_end_time = time.time()

                    df = pd.DataFrame(
                        data=rms,
                        columns=pd.MultiIndex.from_frame(data_header), #Multilevel columns (like a hierarchy)
                        index=pd.to_datetime([t.datetime for t in times_rms], utc=True)
                    )

                    #Resample data to get 1 data point per hour
                    df = df.resample(f"{int(rms_step)}h").median()
                    # Concatenate along columns
                    data_col = df if data_col is None else pd.concat([data_col, df], axis=1)
                    #print(f"RMS computation took:{rms_end_time - rms_start_time:.2f} seconds.")
                    
        #Concatenate along rows
        if data_col is not None:
            df_seism = data_col if df_seism is None else pd.concat([df_seism, data_col], axis=0)
            print(f"Appended RMS data for day: {time_day_str}")

        day_end_time = time.time()
        print(f"Processed day {time_day_str} in {day_end_time - day_start_time:.2f} seconds.")
        
    return df_seism
    

def SaveData(folder_save, file_out, df_seism, df_sea):

    try:
        full_path = f"{folder_save}/{file_out}"
        
        #Open file and save all data in binary format
        with open(full_path, 'wb') as handle:
            for data in [df_seism, df_sea]:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Data saved successfully to:", full_path)
    
    except Exception as e:
        print(f"Error saving data: {e}")

total_start_time = time.time()
   
df_sea, org_shape = LocalSeaAnalysis()

df_seism = LocalRMSAnalysis()

SaveData(folder_save, file_out, df_seism, df_sea)

total_end_time = time.time()

print(f"Processed all data in {total_end_time - total_start_time:.2f} seconds.")