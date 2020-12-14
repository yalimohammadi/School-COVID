#READ this: https://cmu-delphi.github.io/delphi-epidata/api/README.html
# https://cmu-delphi.github.io/covidcast/covidcast-py/html/

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

res = Epidata.covidcast('hospital-admissions', 'smoothed_adj_covid19_from_claims', 'day', 'county', [20201205], 42003)
# print(res['result'], res['message'], len(res['epidata']))
print(res) #TODO: figure out how to get county data

