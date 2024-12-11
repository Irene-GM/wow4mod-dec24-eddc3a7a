import os
import sys
import rpy2
import time
import logging
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from datetime import datetime, timedelta
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


####################
# Global variables #
####################

rowid = 1


pd.options.display.max_columns = 500
pd.options.display.width = 0

header_wow = [ 'Id', 'Site Id', 'Longitude', 'Latitude',
               'Report Date / Time', 'Air Temperature', 'Wet Bulb', 'Dew Point',
               'Mean Sea-Level Pressure', 'Pressure (At Station)', 'Relative Humidity',
               'Rainfall Rate', 'Rainfall Accumulation', 'Wind Speed',
               'Wind Direction', 'Wind Gust', 'Wind Gust Direction', 'geometry']


basic_func_robust = """
         library(robustbase)
         packageVersion("robustbase")
         robust.calc <- function(x){
           x_nonan = na.omit(x)
           qn = Qn(x_nonan, constant=2.21914)
           sqn = s_Qn(x_nonan, mu.too=TRUE)
           loc_median = sqn[1]
           scale_qn = sqn[2]
           return (list(qn=qn, loc_median=loc_median, scale_qn=scale_qn))                       
         }
         """


#############
# Functions #
#############

def read_reference_data_daily(basedir):
    # The difference between this one and the following is that the file
    # structure in the KNMI Data Platform seems to have change, so now for
    # daily files all observations come in a single file, whereas the hourly
    # observations are split into decadal folders.
    dic = defaultdict(object)
    for root, dirs, files in sorted(os.walk(basedir)):
        for file in files:
            path_cur = os.path.join(root, file)
            df = pd.read_csv(path_cur, sep=";", header=0, parse_dates=[1])
            df.index += 1
            # Check if empty
            if df.shape[0] > 0:
                aws_id = df['STN'].iat[0]
                dic[aws_id] = df
    return dic


# ------------------------------------------------------------------------------------ #
# Mechanistic checks
# ------------------------------------------------------------------------------------ #

def check_if_invalid_station(df, invalidst):
    return df["Site Id"].isin(invalidst)

def check_if_incorrect_metadata(df):
    return df["Latitude"] == df["Longitude"]

def check_if_day_mon_insufficient_coverage(df, obsperday, obspermon, lstations):
    # Some utility dates
    cur_date_day = pd.to_datetime(df.iloc[0]["Report Date / Time"], format='%Y%m%d').date()
    cur_date_mon = datetime(cur_date_day.year, cur_date_day.month, 1)

    # The logic here is to cross/intersect the list with all stations with the
    # list of stations in the current dataframe. For the existing ones contributing
    # values in this hourly dataframe, we retrieve its index in the longer list, that
    # is, for 2015-2019 there are 1-966 stations, thus indices.
    sdf = pd.DataFrame.from_records({"Site Id": lstations})
    col_station = sdf[sdf["Site Id"].isin(df["Site Id"])].index.tolist()
    nobs_day = obsperday.iloc[:, col_station].loc[cur_date_day.strftime("%Y-%m-%d")]
    nobs_mon = obspermon.iloc[:, col_station].loc[cur_date_mon.strftime("%Y-%m-%d")]

    # Make decision on whether or not there is an insufficient number of day/month
    # observations for the stations
    sufi_day = nobs_day > 19
    sufi_mon = nobs_mon > 24 * 30

    # We append these columns to the original hourly dataframe, so the order is kept
    # and each original observation gets two extra annotations.
    merged_day = df.merge(sufi_day, how="left", left_on='Site Id', right_on=sufi_day.index)
    merged_mon = merged_day.merge(sufi_mon, how="left", left_on='Site Id', right_on=sufi_mon.index)
    merged_mon.rename(columns={merged_mon.columns[-2]: "sufi_day"}, inplace=True)
    merged_mon.rename(columns={merged_mon.columns[-1]: "sufi_mon"}, inplace = True)

    # Now we return only the sufi_day, sufi_mon columns, as if they were annotations
    # for each observations. In the event this returns a NAN (unknown reasons), we cast
    # this value to False, so the daily/monthly coverage is deemed insufficient.
    sufi = merged_mon[["sufi_day", "sufi_mon"]]
    sufi = sufi.fillna(value=False)
    return sufi



def mechanistic_filter(df, dic):
    # Remember that when the db was built, observations with incorrect
    # timestamps were removed or corrected
    is_inv = ~check_if_invalid_station(df, dic["inv_sta"])
    is_meta_inco = ~check_if_incorrect_metadata(df)
    is_sufi_cove = check_if_day_mon_insufficient_coverage(df, dic["obsperday"], dic["obspermon"], dic["lsta"])
    df_mf = pd.concat([is_inv, is_meta_inco, is_sufi_cove], axis=1, names=["IsOkStation", "IsOkMetadata", "SufiDay", "SufiMon"])
    return df_mf


# ------------------------------------------------------------------------------------- #
# Statistical checks
# ------------------------------------------------------------------------------------- #

def correct_temp_lapse_rate_arr(thedf, sta_ids, elevdf):

    # Extra Jun22: patch 'thedf' name for the merge
    # thedf = thedf.rename(columns={"station_id": "StationId"})

    b = elevdf["Site Id"].isin(sta_ids)
    c = elevdf[b==True]

    # This merge introduces some empty rows for unknown reasons
    m = thedf.merge(c, how='left', on="Site Id")
    tfin = m["Air Temperature"] + 0.0065 * (m["elevation"] - m["elevation"])
    return tfin.round(decimals=4)


def z_score_and_qn_estimator(tcorr):
    pandas2ri.activate()
    # r_base_lib_in_your_system = r"~/R/x86_64-pc-linux-gnu-library/3.6"
    r_base_lib_in_your_system = r"/opt/conda/lib/R/library/"
    tcorr_median = tcorr.median(skipna=True)

    # This might not seem relevant but apparently forces the
    # rpy2 package to actually look where the robust package is
    rob = importr("robustbase", lib_loc= r_base_lib_in_your_system)
    robjects.r(basic_func_robust)
    func_robust = robjects.r["robust.calc"]
    qn, loc, scale = func_robust(tcorr)

    z = (tcorr - tcorr_median)/qn
    z_filter = z.where((z>-2.32) & (z<1.64))
    z_filter[~pd.isna(z_filter)] = True
    z_filter = z_filter.fillna(value=False)

    return z_filter

def check_inrange(value, median, stdev):
    if (median - stdev - stdev) <= value <= (median + stdev + stdev):
        return True
    else:
        return False

def correlation_filter(df):

    # TODO: This part of the quality control requires revisiting, because
    # the original piece is developed for cities (small scale). There, calculating
    # correlations (i.e. 1 vs N) makes sense because the climatic area will not
    # change in space. However for a continental-size dataset, calculating how
    # the measurements of a station compare with the rest does not seem a 
    # reasonable approach. It's comparing how a station in Finland compares
    # with a station in Spain. Plus, I realize now that this type of "median"
    # based filter biases the choice of the median to locations more densely
    # monitored by stations. 

    # Now the filter will be even more "intra-station", because for a given  hour
    # it takes all the measurements produced in that slot and calculate how much
    # an observation deviates from the median of the hour. The cut-off criteria
    # is 1 stdev.     
    groups = list(df.groupby(["Site Id"]))
    for name, group in groups:
        group_median = group["Air Temperature"].median()
        group_mean = group["Air Temperature"].mean()
        group_stdev = group["Air Temperature"].std()
        group["Corr"] = group["Air Temperature"].apply(check_inrange, args=(group_median, group_stdev))      

    df_concat = pd.concat([item[1] for item in groups])
    return df_concat[["Corr"]]


def statistic_filter(df, dic):
    tcorr = correct_temp_lapse_rate_arr(df, dic["lsta"], dic["elev"])
    z_score = z_score_and_qn_estimator(tcorr)
    correlation = correlation_filter(df)
    df_sf = pd.concat([z_score, correlation], axis=1, names=["Z-score", "Corr"])
    return df_sf

def find_level(row):
    if all(True == it for it in [row["IsOkStation"], row["IsOkMetadata"]]):
        if row["Z-score"] == True:
            if all(True == it for it in [row["SufiDay"], row["SufiMon"]]):
                if row["Corr"] == True:
                    return "M4"
                else:
                    return "M3"
            else:
                return "M2"
        else:
            return "M1"
    else:
        return "M0"


def quality_control(df, dic):
    col_rowid_obs = df[["Id"]]

    logging.info("Proceeding with mechanistic filter...")
    flags_meca = mechanistic_filter(df, dic)
    logging.info("Proceeding with statistic filter...")
    flags_stat = statistic_filter(df, dic)

    qc = pd.concat([col_rowid_obs, flags_meca, flags_stat], axis=1, sort=False)
    qc.index = np.arange(1, len(qc) + 1)
    qc = qc.reset_index(drop=False)
    qc.columns = ["rowid_obs", "Id", "IsOkStation", "IsOkMetadata", "SufiDay", "SufiMon", "Z-score", "Corr"]
    qc['Level'] = qc.apply(find_level, axis=1)

    return qc

