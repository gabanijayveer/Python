import pandas as pd

data = {
    'Name': ['JV', 'Annu', 'Peta', 'Liccey','Furry','Muttry','Raj'],
    'Age': [28, 24, 35, 32,25,20,32],
    'City': ['Surat', 'Baroda', 'Anand', 'London','Paris','Mumbai','jaipur']
}

df = pd.DataFrame(data)

print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.dtypes)

df['Salary'] = [70000,52000,74200,85200,65200,45200,95200]

print(df)
print(df.shape)
print(df.dtypes)

df.rename(columns={'City': 'Location'}, inplace=True)
print(df)


filtered_df = df[df['Age'] > 30]
print(filtered_df)

import pandas as pd
# Create a Series from a list
data = [10, 20, 30, 40, 50]
s = pd.Series(data)
print(s)


#  Handling Missing Data

data_with_non={
	'name': ['jimit','Frish',None],
	'age':[25,None,35],
	'city':['surat',None,'baroda']
}
	
df_non = pd.DataFrame(data_with_non)
df_non.fillna({'name':'unknown','age':0,'city':'unknown'},inplace=True)

print(df_non)



data_with_non={
	'name': ['jimit','Frish',None],
	'age':[25,None,35],
	'city':['surat',None,'baroda']
}
	
df_non = pd.DataFrame(data_with_non)
df_non.dropna(inplace=True)
print(df_non)
