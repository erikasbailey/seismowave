import numpy as np
import pandas as pd
from obspy import UTCDateTime
from netCDF4 import Dataset
import glob
import pickle

import time

#Directory containing hindcast map (netCDF4)
local_sea = "Input/Sea data/Data"

#Analysis period
starttime = UTCDateTime('2021-07-01 00:00:00')
endtime = UTCDateTime('2021-12-31 23:59:59')

#file name to save data
pickle_filename = "Processed Data/sea_data_hourly_2021H2.pickle"

#Date format for filenames
dateformat = '%Y%m%d'

#ROI boundaries for the Sicilian Channel Sea
lat1, lat2 = 35.10, 39.0 #latitude range - Minio used lat2=38.09 but this excludes part of the ROI to the nort of sicily
lon1, lon2 = 11.45, 15.87 #Longitude range

#Time intervals to read (1-hour intervals)
nhours = 1
time_period = np.arange(starttime, endtime, 3600 * nhours)  #3600 seconds in an hour

def LocalSeaAnalysis():
    """
    Extracts significant wave height (VHM0) data from netCDF files for a specified time range and ROI.
    """
    #Initialise storage for sea data
    df_sea = {'Y': [], 'time': [], 'lat': None, 'lon': None}
    previous_day_str = None
    # daystart = time.time()

    for time_hour in time_period:
        #Convert time to UTCDateTime format
        time_utc = UTCDateTime(time_hour)
        time_day_str = time_utc.strftime(dateformat)  # Day as 'YYYYMMDD'
        time_str = time_utc.strftime('%H%M')  # Hour-Minute as 'HHMM'

        #Notify when a new day starts
        if previous_day_str and time_day_str != previous_day_str:
            # dayend=time.time()
            # timeday = dayend - daystart
            # print(f"Day processed in seconds: {timeday}")
            print(f"Completed processing for day: {previous_day_str}")
            # daystart = time.time()
            
        previous_day_str = time_day_str

        #Build file pattern for current hour
        file_pattern = f"{local_sea}/{time_str}_{time_day_str}.nc"
        file_list = glob.glob(file_pattern)

        if not file_list:
            print(f"No file found for time {time_str} on day {time_day_str}")
            continue

        for file in file_list:
            with Dataset(file) as f:
                #Extract VHM0 (significant wave height)
                height = f.variables['VHM0'][:] #Read height data
                height = np.ma.filled(height, np.nan) #Replace masked values with NaN

                #Time array
                time_data = pd.to_datetime(f.variables['time'][:], unit='s', utc=True)

                #Create latitude and longitude arrays
                latitude = np.linspace(f.variables['latitude'].valid_min, 
                                       f.variables['latitude'].valid_max, 
                                       np.shape(height)[0])
                longitude = np.linspace(f.variables['longitude'].valid_min, 
                                        f.variables['longitude'].valid_max, 
                                        np.shape(height)[1])

                #Crop latitude and longitude arrays based on ROI
                lat_indices = np.where((latitude >= lat1) & (latitude <= lat2))[0]
                lon_indices = np.where((longitude >= lon1) & (longitude <= lon2))[0]

                #Determine the range of indices for cropping
                mr, Mr = np.min(lat_indices), np.max(lat_indices)
                mc, Mc = np.min(lon_indices), np.max(lon_indices)

                #Crop height data to the ROI
                height_cropped = height[mr:Mr + 1, mc:Mc + 1]

                #Crop latitude and longitude grids
                Lat, Lon = np.meshgrid(latitude[mr:Mr + 1], longitude[mc:Mc + 1], indexing='ij')

                #Append the cropped data to storage
                df_sea['Y'].append(height_cropped)
                df_sea['time'].append(time_data.to_pydatetime())

                #Store lat and lon once (assumes consistent across files)
                if df_sea['lat'] is None or df_sea['lon'] is None:
                    df_sea['lat'] = np.flipud(Lat)
                    df_sea['lon'] = np.flipud(Lon)

    #Stack height data along a new axis
    if df_sea['Y']:
        df_sea['Y'] = np.stack(df_sea['Y'], axis=0)
    else:
        df_sea['Y'] = np.array([]) #Empty array if no data was found

    #Convert time list to pandas Series
    df_sea['time'] = pd.Series(df_sea['time'])

    #Verify the shape of the extracted matrix
    org_shape = df_sea['Y'].shape
    print(f"OK! Sea wave time series were extracted. Shape: {org_shape}")

    return df_sea, org_shape

#Run the analysis
df_sea, org_shape = LocalSeaAnalysis()

#Save the processed data to a pickle file
with open(pickle_filename, 'wb') as pickle_file:
    pickle.dump(df_sea, pickle_file)
    print(f"Data successfully written to {pickle_filename}")

# Print summary
print(f"Extracted sea wave matrix shape: {org_shape}")
print(f"Number of time records: {len(df_sea['time'])}")
