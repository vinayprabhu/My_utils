def ts_to_time(ts):
    # Convert device timestamps to datetime objects
    utc_dt=datetime.datetime(year=2001,month=1,day=1)+datetime.timedelta(microseconds=np.float(ts))
    return utc_dt
######
# Rounding the device timestamp to the nearest 'T' minute interval
T_minutes=30
offset_T=T_minutes*60*1e9  # T minutes in nanoseconds 
df_test['time_rnd']=pd.to_datetime(((df_test.timestamp.apply(ts_to_time).astype(np.int64) // offset_T + 1 ) * offset_T))


