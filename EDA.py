#!/usr/bin/python3

# data set obtained from:
# kaggle.com/kaggle/sf-salaries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In this example we interested in EDA on the salaries data



path = 'Salaries/Salaries.csv'
df = pd.read_csv(path)

print(df.head())
print('~~~~~~~~~~~~~')
print(df.shape)
print('~~~~~~~~~~~~~')
print(df.columns)
print('~~~~~~~~~~~~~')
print(df.isna().sum())
print ('~~~~~~~~~~~~')

# Drop columns we not interested in

df = df.drop(['Id','Notes','Status'],axis=1)

# Convert jobtitles to uppercase
# This is convernient for later when we select a particular
# dataset to work with. It will help focus the business question
# into a theme industry since the job titles range a lot.

df['JobTitle'] = df['JobTitle'].str.upper()

print(df.head())
print('~~~~~~~~~~~~')

# Check variables

print(df.JobTitle)

# Check data type

print(df.dtypes)
print('~~~~~~~~~~~~')

# Substitute unwanted values and convert data type
# There is some incorrent entries in payment values
# If later we are to build ML model to work further 
# on this data, incorrent entries must first be fixed

df['BasePay'] = df['BasePay'].replace('Not Provided', np.nan)
df['BasePay'] = df['BasePay'].replace(0, np.nan)
df.BasePay = df.BasePay.astype('float64')

df['OvertimePay'] = df['OvertimePay'].replace('Not Provided', np.nan)
df['OvertimePay'] = df['OvertimePay'].replace(0, np.nan)
df.BasePay = df.OvertimePay.astype('float64')

df['OtherPay'] = df['OtherPay'].replace('Not Provided', np.nan)
df['OtherPay'] = df['OtherPay'].replace(0, np.nan)
df.BasePay = df.OtherPay.astype('float64')

df['Benefits'] = df['Benefits'].replace('Not Provided', np.nan)
df['Benefits'] = df['Benefits'].replace(0, np.nan)
df.BasePay = df.Benefits.astype('float64')

print('Describe data')
print(df.OtherPay)
print(df.dtypes)

# Check for Duplicates based on job title

df_duplicated = df[df['JobTitle'].duplicated()]

print(df_duplicated.head())
print('~~~~~~~~~~~~')
print(df_duplicated.shape)

# Check for Uniqueness based on job title

df_unique = df['JobTitle'].unique()

print('~~~~~~~~~~~~')
print(df_unique)

# Select only certain job titles for analysis
# We interested in jobs in the health sector

df1 = df[df['JobTitle'].str.contains('ASSISTANT MEDICAL EXAMINER', regex=False)]
df2 = df[df['JobTitle'].str.contains('SENIOR PHYSICIAN SPECIALIST', regex=False)]
df3 = df[df['JobTitle'].str.contains('NURSING SUPERVISOR', regex=False)]
df4 = df[df['JobTitle'].str.contains('ANESTHETIST', regex=False)]
df5 = df[df['JobTitle'].str.contains('EPIDEMIOLOGIST', regex=False)]

selected_jobs = ['ASSISTANT MEDICAL EXAMINER',
        'SENIOR PHYSICIAN SPECIALIST',
        'NURSING SUPERVISOR',
        'ANETHETIST',
        'NURSE MANAGER',
        'SPECIAL NURSE',
        'NURSE MIDWIFE',
        'NURSE PRACTITIONER',
        'PATIENT CARE ASSISTANT',
        'HEALTH WORKER 3']

df6 = df[df.JobTitle.str.contains('|'.join(selected_jobs))]
print(df1)

selected_professions = ['CHEMIST',
        'BIOLOGIST',
        'EPIDEMIOLOGIST',
        'NUTRITIONIST']

df7 = df[df.JobTitle.str.contains('|'.join(selected_professions))]

# Try grouping by base pay

df8 = df7.groupby(['TotalPay','JobTitle']).count()['EmployeeName']
print('~~~~~~~~~~~~~ Grouped by ~~~~~~~~~~~~~~~~~~~')
print(df8)

# Visualize

#sns.set(font_scale=1.0)
fig1 = plt.figure(figsize=(20,5))
chart1 = sns.barplot(x='JobTitle', 
        y='TotalPay', 
        data=df5,
        hue='Year',
        palette='Set1')

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=5)
chart1.set_ylim(df5.TotalPayBenefits.min(),df5.TotalPayBenefits.max())

fig2 = plt.figure(figsize=(15,5))
chart2 = sns.boxplot(data=df5, 
        x='JobTitle',
        y='TotalPay',
        hue='Year')

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=5)
#chart2.set_ylim(df1.TotalPayBenefits.min(),df1.TotalPayBenefits.max())

fig3 = plt.figure(figsize=(15,5))
chart3 = sns.countplot(data=df5[df5['Year']==2011], 
#        x='JobTitle',
        x='TotalPay',
        hue='JobTitle')

chart3.set_xticklabels(chart3.get_xticklabels(), rotation=15)

# Check for correlation between columns

corr = df.corr()

fig4 = plt.figure()
chart4 = sns.heatmap(corr,annot=True)

fig5,ax = plt.subplots(figsize=(15,7))
df8.unstack().plot(ax=ax)

plt.show()
