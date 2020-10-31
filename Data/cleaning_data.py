import pandas as pd
# for the first time, run pip install xlrd


sample_path=r"Test_Results_Locations_LAUSD_delivered_10302020.xlsx"

raw_data = pd.read_excel (sample_path)

df = pd.DataFrame(raw_data, columns= ["index",'Unique Person Num','Tester Status','Test Date','Result',
                                      'Age','Residence','Site Type','Cost Center Code','School or Site Name'])
# print (df) #all data

is_positive= df["Result"]=="Positive"

positive_testers=df[is_positive] #get rows that got positive result
print(positive_testers)