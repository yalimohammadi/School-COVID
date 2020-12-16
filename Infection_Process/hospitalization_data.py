#READ this: https://cmu-delphi.github.io/delphi-epidata/api/README.html
# https://cmu-delphi.github.io/covidcast/covidcast-py/html/
import seaborn as sns
import matplotlib.pyplot as plt
from delphi_epidata import Epidata
from datetime import date

#guide: https://healthdata.gov/covid-19-reported-patient-impact-and-hospital-capacity-state-data-dictionary
res = Epidata.covid_hosp('MA', 20200510)
print(res)
data=res['epidata'][0]
print(res['result'], res['message'], data['inpatient_bed_covid_utilization_numerator'])



# Santa clara FIPS code: 06085
# res =  Epidata.covidcast("fb-survey", "smoothed_cli", date(2020, 5, 1), date(2020, 5, 7),
#                         "county")
# # print(res['result'], res['message'], len(res['epidata']))
# print(res)

res = Epidata.covidcast('hospital-admissions', 'smoothed_covid19_from_claims', 'day', 'state', [20201105], "CA")
print(res['result'], res['message'])
res = Epidata.covidcast('hospital-admissions', 'smoothed_covid19_from_claims', 'day', 'county', [Epidata.range(20201101, 20201128)], "06037")
data= res['epidata']

def get_covid_data(epidata):
    for dic in epidata:
        dic["value"]

print(res) #TODO: figure out how to get county data: URL  https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html

