import pandas as pd
import numpy as np

df = pd.read_csv("credit_card_transactions-ibm_v2.csv")
zip_df = pd.read_csv('USZipsWithLatLon_20231227.csv')
city_df = pd.read_csv("worldcities.csv")
user_df = pd.read_csv("sd254_users_modified.csv")
card_df = pd.read_csv("sd254_cards_modified.csv")

temp_dt = pd.to_datetime(pd.DataFrame({
    'year': df['Year'],
    'month': df['Month'],
    'day': df['Day'],
    'hour': df['Time'].str.split(':').str[0].astype(int),
    'minute': df['Time'].str.split(':').str[1].astype(int)
}))
df.insert(2, 'unixtime', (temp_dt.astype('int64') // 10**9))

df['msin'] = np.sin((df['Month']-1)*2*np.pi/12).astype('float32')
df['mcos'] = np.cos((df['Month']-1)*2*np.pi/12).astype('float32')

denom = np.full(len(df), 31, dtype=np.int16)
denom[df['Month'].isin([4, 6, 9, 11])] = 30
is_feb = df['Month'] == 2
is_leap = (df['Year'] % 4 == 0)
denom[is_feb & is_leap] = 29
denom[is_feb & ~is_leap] = 28
df['dsin'] = np.sin(2*np.pi*(df['Day']-1)/denom).astype('float32')
df['dcos'] = np.cos(2*np.pi*(df['Day']-1)/denom).astype('float32')

weekday = temp_dt.dt.weekday
df['wsin'] = np.sin(2*np.pi*weekday/7).astype('float32')
df['wcos'] = np.cos(2*np.pi*weekday/7).astype('float32')
df['is_weekend'] = (weekday >= 5).astype('int64')

time_split = df['Time'].str.split(':')
minutes_total = time_split.str[0].astype(int) * 60 + time_split.str[1].astype(int)
df['tsin'] = np.sin(minutes_total * 2 * np.pi / 1440).astype('float32')
df['tcos'] = np.cos(minutes_total * 2 * np.pi / 1440).astype('float32')

df['Amount'] = df['Amount'].str.replace('$', '').astype('float32')

df['Chip_Online'] = (df['Use Chip'] == 'Online Transaction').astype(int)
df['Chip_Swipe'] = (df['Use Chip'] == 'Swipe Transaction').astype(int)

df['Is Fraud?'] = (df['Is Fraud?'] == 'Yes').astype(int)

merchant_counts = df['Merchant Name'].value_counts()
low_freq_merchants = merchant_counts[merchant_counts < 244].index
df.loc[df['Merchant Name'].isin(low_freq_merchants), 'Merchant Name'] = 'Others'

error_list = ['Insufficient Balance', 'Bad PIN', 'Technical Glitch', 'Bad Card Number', 'Bad CVV', 'Bad Expiration', 'Bad Zipcode']

for err in error_list:
    df[f'Error_{err}'] = df['Errors?'].str.contains(err, na=False).astype(int)

df = df.merge(zip_df[['zipcode', 'latitude', 'longitude']], how='left', left_on='Zip', right_on='zipcode')

df.loc[df['Merchant City'] == 'ONLINE', ['latitude', 'longitude']] = 0

df = df.merge(city_df[['city_ascii', 'country', 'lat', 'lng']], how='left', left_on=['Merchant City', 'Merchant State'], right_on=['city_ascii', 'country'])

df['Merchant_Lat'] = (df['lat'].combine_first(df['latitude'])).astype('float32')
df['Merchant_Lng'] = (df['lng'].combine_first(df['longitude'])).astype('float32')

drop_cols = ['Month', 'Day', 'Time', 'Zip', 'zipcode', 'Errors?', 'city_ascii', 'country', 'lat', 'lng', 'Use Chip', 'latitude', 'longitude']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

df = df.merge(user_df, how='left', left_on='User', right_on='User_index')
df.drop(columns=['User_index', 'User_Current Age'], inplace=True)

df['Home_Lat'] = (df['Home_Latitude']).astype('float32')
df['Home_Lng'] = (df['Home_Longitude']).astype('float32')
df.drop(columns=['Home_Latitude', 'Home_Longitude'], inplace=True)

df = df.merge(card_df, how='left', left_on=['User', 'Card'], right_on=['Card_User', 'Card_Index'])
df.drop(columns=['Card_User', 'Card_Index'], inplace=True)

df['Card_Ex_Month'] = df['Card_Expires'].str.split('/').str[0].astype(int)
df['Card_Ex_Year'] = df['Card_Expires'].str.split('/').str[1].astype(int)
df.drop(columns=['Card_Expires'], inplace=True)

df['Card_Open_Month'] = df['Card_Open'].str.split('/').str[0].astype(int)
df['Card_Open_Year'] = df['Card_Open'].str.split('/').str[1].astype(int)
df.drop(columns=['Card_Open'], inplace=True)

df['Card_Brand_Master'] = (df['Card_Brand'] == 'Mastercard').astype(int)
df['Card_Brand_Amex'] = (df['Card_Brand'] == 'Amex').astype(int)
df['Card_Brand_Discover'] = (df['Card_Brand'] == 'Discover').astype(int)
df.drop(columns=['Card_Brand'], inplace=True)

df['Card_Type_Debit'] = (df['Card_Type'] == 'Debit').astype(int)
df['Card_Type_Credit'] = (df['Card_Type'] == 'Credit').astype(int)
df.drop(columns=['Card_Type', 'Merchant City'], inplace=True)

df['Merchant State'] = df['Merchant State'].fillna('ONLINE')

df['Distance_Home'] = ((df['Merchant State'] != 'ONLINE').astype('float32'))*2*6371*(np.arcsin((((np.sin((df['Merchant_Lat']*np.pi/180-df['Home_Lat']*np.pi/180)/2))**2) + ((np.cos(df['Merchant_Lat']*np.pi/180)*np.cos(df['Home_Lat']*np.pi/180))*((np.sin((df['Merchant_Lng']*np.pi/180-df['Home_Lng']*np.pi/180)/2))**2)))**0.5))

df.sort_values(by=['User', 'unixtime'], inplace=True)

df['last_off_Lat'] = df.groupby('User')['Merchant_Lat'].transform(
    lambda x: x.where(df.loc[x.index, 'Merchant State'] != 'ONLINE', np.nan).ffill().shift(1)
)

df['last_off_Lng'] = df.groupby('User')['Merchant_Lng'].transform(
    lambda x: x.where(df.loc[x.index, 'Merchant State'] != 'ONLINE', np.nan).ffill().shift(1)
)

df['last_off_State'] = df.groupby('User')['Merchant State'].transform(
    lambda x: x.where(df.loc[x.index, 'Merchant State'] != 'ONLINE', np.nan).ffill().shift(1)
)

df['last_off_Time'] = df.groupby('User')['unixtime'].transform(
    lambda x: x.where(df.loc[x.index, 'Merchant State'] != 'ONLINE', np.nan).ffill().shift(1)
)

df['last_off_Lat'] = df['last_off_Lat'].fillna(0)
df['last_off_Lng'] = df['last_off_Lng'].fillna(0)
df['last_off_State'] = df['last_off_State'].fillna('ONLINE')
df['last_off_Time'] = (df['last_off_Time'].fillna(1)).astype('int64')

df['Distance_Delta'] = (((df['Merchant State'] != 'ONLINE') & (df['last_off_State'] != 'ONLINE')).astype('float32'))*2*6371*(np.arcsin((((np.sin((df['Merchant_Lat']*np.pi/180-df['last_off_Lat']*np.pi/180)/2))**2) + ((np.cos(df['Merchant_Lat']*np.pi/180)*np.cos(df['last_off_Lat']*np.pi/180))*((np.sin((df['Merchant_Lng']*np.pi/180-df['last_off_Lng']*np.pi/180)/2))**2)))**0.5))
df['velocity_kph'] = (df['Distance_Delta'] / ((df['unixtime']-df['last_off_Time'])/3600)).astype('float32').replace([np.inf, -np.inf], 0).fillna(0)
df['supersonic'] = (df['velocity_kph'] > 1200).astype('int64')

user_mean = df.groupby('User')['Amount'].transform('mean')
user_max = df.groupby('User')['Amount'].transform('max')
df['ratio_mean'] = (df['Amount'] / user_mean).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
df['ratio_max'] = (df['Amount'] / user_max).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')

df['cum_mean'] = df.groupby('User')['Amount'].transform(lambda x: x.expanding().mean())
df['cum_max'] = df.groupby('User')['Amount'].transform(lambda x: x.expanding().max())
df['last_cum_mean'] = df.groupby('User')['cum_mean'].shift(1)
df['last_cum_max'] = df.groupby('User')['cum_max'].shift(1)
df['ratio_cumulative_max'] = (df['Amount'] / df['last_cum_max']).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
df['ratio_cumulative_mean'] = (df['Amount'] / df['last_cum_mean']).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
df.drop(columns=['cum_mean', 'last_cum_mean', 'cum_max', 'last_cum_max'], inplace=True)

df['ratio_to_card_limit'] = (df['Amount']/df['Card_Limit']).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
df['ratio_to_u_income'] = (df['Amount']/df['User_Income']).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
df['ratio_to_z_income'] = (df['Amount']/df['Home_Zipcode Income']).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')

df.to_csv('done.csv')