import seaborn as sns
import matplotlib.pyplot as plt
from delphi_epidata import Epidata
from datetime import date
import pandas as pd
import numpy as np
from datetime import date, timedelta


def allMondays():
    d = pd.to_datetime("20200914", format='%Y%m%d')
    max_date = pd.to_datetime("20201205", format='%Y%m%d')
    year = 2020
    #     d += timedelta(days = 6 - d.weekday())  # First Sunday
    while d < max_date:
        yield d
        d += timedelta(days=7)

def get_covid_data(epidata):
    values = []
    for dic in epidata:
        v = dic["value"]
        values.append(v)
    return values


def get_covid_data_weekly(epidata):
    values = dict()
    for date in allMondays():
        values[date] = 0
    for dic in epidata:
        my_day = pd.to_datetime(dic["time_value"], format='%Y%m%d')
        off_set = my_day.weekday()
        monday = my_day - timedelta(off_set)

        v = dic["value"]
        values[monday] += v
    for date in allMondays():
        values[date] /= 7.
    return values

res = Epidata.covid_hosp('MA', 20200510)

# Santa clara FIPS code: 06085

res = Epidata.covidcast('hospital-admissions', 'smoothed_covid19_from_claims', 'day', 'county',
                        [Epidata.range(20200914, 20201204)], "06037")  # 06037 is lA county FIPS code

data = res['epidata']
covid_hosp_cases_weekly = get_covid_data_weekly(data)
covid_hosp_cases = get_covid_data(data)

res = Epidata.covidcast('usa-facts', 'confirmed_incidence_num', 'day', 'county', [Epidata.range(20200914, 20201204)],
                        "06037")
covid_confirmed_cases = get_covid_data(res['epidata'])
covid_confirmed_cases_weekly = get_covid_data_weekly(res['epidata'])


plt.plot(pd.date_range(start="2020-09-14", end="2020-12-04"),[v*1000 for v in covid_hosp_cases],label="1000 x percentage of hospital confirmed cases")
plt.plot(pd.date_range(start="2020-09-14", end="2020-12-04"),covid_confirmed_cases,label="confirmed cases- USA facts")
plt.legend()
plt.title("Los Angeles County")
plt.show()



# age group labels
age_def={ 1: "0-4", 2:"5-17", 3: "18-29", 4: "30-39", 5:"40-49", 6: "50-64", 7:"65-74", 8: "75-84", 9:"85+"}
Age_groups_risks = np.array([1/4.,1/9.,1.,2.,3.,4.,5.,8.,13.])
Age_group_hospitalization = Age_groups_risks/sum(Age_groups_risks)


# https://www.statista.com/statistics/1122354/covid-19-us-hospital-rate-by-age/
Age_group_hospitalization_to_cases = [100/(1.84),100/(1.06),1000./7.85, 100/12.1, 100/17.6,100/26.6,100/36.1,100/57.2,100/86.5]

def find_age_group(age):
    if age in range(0,5):
        return 1
    if age in range(5,18):
        return 2
    if age in range(18,30):
        return 3
    if age in range(30,40):
        return 4
    if age in range(40,50):
        return 5
    if age in range(50,65):
        return 6
    if age in range(54,75):
        return 7
    if age in range(75,85):
        return 8
    if age>=85:
        return 9
    return -1

age_group_mapping= dict()
for i in range(1,101):
    age_group_mapping[i]=find_age_group(i)



hosp_by_age=dict()
for i in range(1,10):
    hosp_by_age[i]= [v*Age_group_hospitalization[i-1] for v in covid_hosp_cases_weekly.values()]
    plt.plot(list(covid_hosp_cases_weekly.keys()),hosp_by_age[i],label=age_def[i])
plt.legend()
plt.show()