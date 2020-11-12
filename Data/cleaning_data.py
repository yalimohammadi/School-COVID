import pandas as pd
import numpy as np
# for the first time, run pip install xlrd


sample_path=r"Test_Results_Locations_LAUSD_delivered_10302020.xlsx"

raw_data = pd.read_excel (sample_path)

df = pd.DataFrame(raw_data, columns= ["index",'Unique Person Num','Tester Status','Test Date','Result',
                                      'Age','Residence','Site Type','Cost Center Code','School or Site Name'])
# print (df) #all data
# df.fillna(0, inplace=True)


df['Age'] = df['Age'].fillna(0)
df['Age'] = df['Age'].astype(int)

# df['Age'] = df['Age'].replace(r'\s+', np.nan, regex=True)


is_positive= df["Result"]=="Positive"

positive_testers=df[is_positive] #get rows that got positive result



elementary_age=p=[5,6,7,8,9,10]
middleschool_age=[11,12,13]
highschool_age=[14,15,16,17]
positive_testers.iloc(1)
elem_students= positive_testers[positive_testers["Age"] in elementary_age]
print(positive_testers)
print(elem_students)