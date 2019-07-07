# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:05:21 2019

@author: diana
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from data_cleaning import *

deal_view = pd.read_csv(r"data\deal_view.csv")
deal_view = deal_view.drop_duplicates()

for col in deal_view.columns:
    print("missing value for " + col + " is " + str(deal_view[col].isnull().sum()))
    print("there are " + str(deal_view[col].nunique()) + " unique values in this column.")
    
useless_col = ['columns that are not useful for this analysis']
deal_view = deal_view.drop(useless_col, axis =1)

context_col = ['columns related to campaign context']
context_unique = pd.DataFrame()
for col in context_col:
    col_unique = deal_view[col].unique()
    context_unique[col] = pd.Series(col_unique)
    
context_unique.to_csv(r"data\unique_context.csv")

deal_view['receipt_at'] = pd.to_datetime(deal_view['receipt_at'], errors='coerce')
deal_view['receipt_date'] = deal_view['receipt_at'].dt.date
deal_view['receipt_day_of_week'] = deal_view['receipt_at'].dt.weekday

view_by_day = pd.DataFrame(deal_view.groupby(['receipt_day_of_week'])['user_id'].count()).reset_index()
view_by_day = view_by_day.rename(columns={"user_id":"number_of_views"})
ax = view_by_day.plot.bar(x='receipt_day_of_week', y='number_of_views', rot=0)

view_by_user = pd.DataFrame(deal_view.groupby(['user_id'])['receipt_at'].count()).reset_index()
view_by_user = view_by_user.rename(columns={"receipt_at":"number_of_views"})
fig, ax = plt.subplots(figsize=(12,9)) 
bins = np.linspace(1,15,100)
plt.hist(view_by_user['number_of_views'], bins, alpha=0.75, label = 'Deal Views per User')
ax.set_xlabel('Number of Deal Views', fontsize=15)
ax.set_ylabel('Number of Users', fontsize=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
plt.savefig('deal views by user.png')

dataTypes, manager, deal, activity, users = loadData(r"data\anonymized-data.xlsx")
clean_user, user_variables = cleanData_user(users)
user_activity_des = user_Analysis(clean_user)
clean_activity, activity_variables = cleanData_activity(activity)
clean_deal, deal_variables = cleanData_deal(deal)
clean_manager, manager_variables = cleanData_manager(manager)
clean_add_deal = cleanData_additionaldeal(add_deal)
deal = merge_deal(clean_deal, clean_add_deal)


view_by_user = view_by_user.merge(clean_user[['UserId','originDate','priorActivity','inactive']], 
                                  how='left', left_on=['user_id'], right_on=['UserId'])
view_by_user = view_by_user.drop(['UserId'], axis=1)
view_by_user = view_by_user.dropna(subset=['originDate'])
view_by_user['lifetime'] = (view_by_user['priorActivity'] - view_by_user['originDate']).dt.days
view_by_user['days_per_view'] = np.where(view_by_user['inactive']==0, 
            np.round(view_by_user['lifetime']/view_by_user['number_of_views'],3),2100)

fig, ax = plt.subplots(figsize=(12,9)) 
bins = np.linspace(0,500,100)
active_cond = (view_by_user['inactive']==0) & (view_by_user['number_of_views']>=2)
plt.hist(view_by_user[active_cond]['days_per_view'], bins, alpha=0.8, 
         label = 'Days Elapsed between Deal Views')
ax.set_xlabel('Days Elapsed', fontsize=15)
ax.set_ylabel('Number of Users', fontsize=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
plt.savefig('days elapsed between deal views_active users.png')


pairing = cross_join(deal, clean_user)
pairing = pairing.merge(deal_view[['user_id','receipt_date','receipt_day_of_week']], 
                        how='left', left_on=['UserId'],
                        right_on=['user_id']).drop(['user_id'],axis=1)
pairing = pairing.drop_duplicates()
pairing = pairing.merge(clean_activity[['dealId','UserId','Submit']], 
                        how='left', left_on=['dealId', 'UserId'],
                        right_on=['dealId','UserId']).drop(['dealId'],axis=1)
pairing = pairing.drop_duplicates()
view_before_deal_end = (pairing['receipt_date'] < pairing['deal_end'])
view_after_deal_start = (pairing['receipt_date'] > pairing['deal_start'])

pairing = pairing[view_before_deal_end]
pairing = pairing.drop_duplicates()
view_per_pairing = pd.DataFrame(pairing.groupby(['dealId','UserId'])['receipy_date'].count()).reset_index()
view_per_pairing = view_per_pairing.rename(columns={"receipt_date":"total_view_so_far"})
pairing = cross_join(deal, clean_user)
pairing = pairing.merge(view_per_pairing, how='left', on=['dealId','UserId'])
pairing['total_view_so_far'] = pairing['total_view_so_far'].fillna(0)



