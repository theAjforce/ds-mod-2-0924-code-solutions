import pandas as pd
import numpy as np

df = pd.read_json('Beauty.json',lines=True)

overall_groups = df.groupby('asin').agg({'overall':[np.size,np.sum,np.mean]})

popular_products = overall_groups.sort_values(('overall','sum'),ascending=False)

def pop_rec(user,popular_info,user_info):
    purchased = user_info.loc[user_info.reviewerID==user,'asin']
    recs = popular_info.drop(index=purchased)
    return recs.index[:5]

recs = pop_rec('A3SFRT223XXWF7',popular_products,df)
#recs = Index(['B000URXP6E', 'B0009RF9DW', 'B000FI4S1E', 'B00W259T7G', 'B0010ZBORW'], dtype='object', name='asin')


df['next_item'] = df.groupby('reviewerID')['asin'].shift(-1)

def next_item(df,user):
  last_item = list(df[df.reviewerID==user]['asin'])[-1]
  purchased = list(df.loc[df.reviewerID==user,'asin'])
  next_item_list = list(df.loc[df.asin==last_item,'next_item'].value_counts().index)
  next_item_list_final = [item for item in next_item_list if item not in purchased]
  next_items = next_item_list_final[:5]
  return next_items

next = next_item(df,'A3J034YH7UG4KT')
#next = ['B01E7UKR38']