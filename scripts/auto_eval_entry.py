import datetime
import os
import copy

fy4_time_interval_list = [
                             15, 45, 60, 45, 15
                         ] * 8

now = datetime.datetime.now()
today = datetime.datetime(now.year, now.month, now.day)
yesterday = datetime.datetime(now.year, now.month, now.day) - datetime.timedelta(days=1)
dt = copy.deepcopy(yesterday)
idx = 0
while dt < today:
    command = dt.strftime("/home/fy4/miniconda3/bin/python /FY4COMM/NBCLM/metesatpy/scripts/auto_eval.py %Y%m%d_%H%M")
    dt += datetime.timedelta(minutes=fy4_time_interval_list[idx])
    idx += 1
    # print(command)
    os.system(command)
