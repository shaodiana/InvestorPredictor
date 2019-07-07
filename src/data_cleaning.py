'''
some variables names are altered to protect company data and may have inconsistent names.
'''
import datetime as dt
import pandas as pd
import numpy as np
from uszipcode import SearchEngine

US_state = ['Alabama - AL', 'Alaska - AK', 'Arizona - AZ', 'Arkansas - AR', 'California - CA',
            'Colorado - CO', 'Connecticut - CT', 'DC', 'Delaware - DE','Florida - FL','Georgia - GA',
            'Hawaii - HI', 'Idaho - ID','Illinois - IL','Indiana - IN','Iowa - IA','Kansas - KS',
            'Kentucky - KY','Louisiana - LA','Maine - ME','Maryland - MD','Massachusetts - MA',
            'Michigan - MI','Minnesota - MN','Mississippi - MS','Missouri - MO','Montana - MT',
            'Nebraska - NE','Nevada - NV','New Hampshire - NH','New Jersey - NJ','New Mexico - NM',
            'New York - NY','North Carolina - NC','North Dakota - ND','Ohio - OH','Oklahoma - OK',
            'Oregon - OR','Pennsylvania - PA','Rhode Island - RI','South Carolina - SC','South Dakota - SD',
            'Tennessee - TN','Texas - TX','Utah - UT','Vermont - VT','Virginia - VA','Washington - WA',
            'West Virginia - WV','Wisconsin - WI','Wyoming - WY',]

US_state1 = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
            'Colorado', 'Connecticut', 'DC', 'Delaware','Florida','Georgia',
            'Hawaii', 'Idaho', 'Illinois','Indiana','Iowa','Kansas',
            'Kentucky','Louisiana','Maine','Maryland','Massachusetts',
            'Michigan','Minnesota','Mississippi','Missouri','Montana',
            'Nebraska','Nevada','New Hampshire','New Jersey','New Mexico',
            'New York','North Carolina','North Dakota','Ohio','Oklahoma',
            'Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota',
            'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
            'West Virginia','Wisconsin','Wyoming',]

US_state2 = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL",
            "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
            "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
            "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
            "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

def loadData(file):
    '''
    read in user data, return user data and dataTypeas dataframe, and
    variables names in a list.
    '''
    xlsx = pd.ExcelFile(file)
    res = len(xlsx.sheet_names)
    print("There are " + str(res) + " worksheets in this xlsx file.")
    data = []
    dataTypes = []
    for i in range(0, res):
        data_i = pd.read_excel(file, sheet_name = i)
        data.append(data_i)
        print("Worksheet " + str(i+1) + " read into DataFrame, with", end=' ')
        print(str(data_i.shape[0]) + " observations, and " + str(data_i.shape[1]) + " variables.")
        dataTypes_i = pd.DataFrame(data_i.dtypes).reset_index().rename(columns={"index":"variable", 0:"type"})
        print("dataTypes collected for worksheet " + str(i+1))
        dataTypes.append(dataTypes_i)
    return pd.concat(dataTypes), data[0], data[1], data[2], data[3]

def loadData_additionaldeal(file):
    '''
    read in the additional deal data
    '''
    add_deal = pd.read_csv(file)
    return add_deal

def replaceString(originalString, toBeReplaced, newString):
    '''
    replace multiple substrings with a new string
    '''
    for substring in toBeReplaced:
        if substring in originalString:
            originalString = originalString.replace(substring, newString)
    return originalString

def cleanData_user(df):
    '''
    Clean user data.
    Time stamp variables: keep date only.
    zipcode: keep first five digits only.
    Region: create a new region variable that matches user input Region to the most similar
            states in the US as two-letter abbreviations.
    variables end with "InCents": convert into thousand dollar, rename to "CAT",
                                  for activity amount in thousand $.
    variables start with "LifetimeAverageDaysBetween": rename to "LADB"
    '''
    timeFeature = ['list of features that are timestamps']
    for col in timeFeature:
        df[col] = pd.to_datetime(df[col], errors ='coerce').dt.date
        newTime = replaceString(col, ['Timestamp','Utc'], '')
        df = df.rename(columns={col:newTime})

    dollarFeature = df.columns[df.columns.str.endswith('InCents')].tolist()
    for col in dollarFeature:
        df[col] = np.round(df[col]/100000, 2)
        newDollar = replaceString(col, ['activitytedAmountInCents','activityAmountInCents'], 'CAT')
        df = df.rename(columns={col:newDollar})

    lengthFeature = df.columns[df.columns.str.startswith('LifetimeAverageDaysBetween')].tolist()
    for col in lengthFeature:
        df = df.rename(columns={col: col.replace('LifetimeAverageDaysBetween','LADB')})

    from fuzzywuzzy import process
    new_region = pd.DataFrame(columns=['old_region','new_region1','score1','new_region2','score2'])
    old_region = df.Region.dropna().unique().tolist()
    for i in range(0, len(old_region)):
        new_region.loc[i] = [old_region[i]] + list(process.extractOne(old_region[i], US_state1)) + \
                      list(process.extractOne(old_region[i], US_state2))

    mask1 = (new_region['old_region'].str.len()>= 2)
    new_region = new_region.loc[mask1]
    new_region = new_region.reset_index(drop=True)

    new_region['new_region3'] = pd.Series()
    for i in range(0, len(new_region)):
        if new_region.loc[i]['score1'] >= new_region.loc[i]['score2']:
            new_region.loc[i,'new_region3'] = new_region.loc[i,'new_region1']
        else:
            new_region.loc[i,'new_region3'] = new_region.loc[i,'new_region2']

    states = pd.DataFrame(list(zip(US_state1, US_state2)), columns =['long_state', 'short_state'])
    new_region = new_region.merge(states, how='left', left_on='new_region3', right_on='long_state')

    df['inactive'] = 0
    df.loc[df['priorActivity']==df['originDate'],'inactive'] = 1

    for i in range(0, len(new_region)):
        if len(new_region.loc[i,'new_region3']) == 2:
            new_region.loc[i,'new_region3'] = new_region.loc[i,'new_region3']
        else:
            new_region.loc[i,'new_region3'] = new_region.loc[i,'short_state']

    df = df.merge(new_region[['old_region','new_region3']], how='left',
                  left_on='Region', right_on='old_region').drop(['old_region'],axis=1).\
                  rename(columns={"new_region3":"new_region"})

    df['zipcode'] = pd.Series()
    for i in range(0, len(df)):
        df.loc[i, 'zipcode'] = str(df.loc[i,'zipCode']).split('-')[0]
        if len(df.loc[i,'zipcode']) != 5:
            df.loc[i,'zipcode'] = 'nan'
        if df.loc[i,'zipcode'].isdigit() == False:
            df.loc[i,'zipcode'] = np.nan
    
    variables = list(df.columns)
    return df, variables

def user_Analysis(clean_df):
    '''
    Run analysis of the clean version of the user data.
    1. Count number of users with non-missing region
    2. Count number of users with non-missing and 5-digit zipcode
    3. Count how many users' last activity is the same as their account create date, i.e. inactive
    4. Describe users' total activity and total CAT, total complete activitys, total complete CAT.
    5. Describe users' LADBCompleteactivity, LADBCompleteactivityAfterRepeat
    6. Describe users' FirstCompleteCAT, AverageCompleteCAT,
    '''
    user_with_region = clean_df.new_region.notnull().sum()
    region_pct = np.round(user_with_region/len(clean_df) * 100, 2)
    print(str(region_pct)+" percentage of the users have region information.")
    user_with_zip = clean_df.zipcode.notnull().sum()
    zip_pct = np.round(user_with_zip/len(clean_df) * 100, 2)
    print(str(zip_pct)+ " percentage of the users have zipcode information.")
    clean_df['inactive'] = 0
    clean_df.loc[clean_df['LastActivity']==clean_df['CreateDate'],'inactive'] = 1
    inactive_pct = np.round(clean_df.inactive.sum()/len(clean_df)*100, 2)
    print(str(inactive_pct) + " percentage of the users are inactive,", end=' ')
    print("i.e. their last activity is on the same day as they register.")

    clean_df['first_activity'] = (clean_df['FirstSubmission'] - clean_df['originDate']).apply(lambda x: x.days)
    clean_df['scnd_activity'] = (clean_df['SecondSubmission'] - clean_df['FirstSubmission']).apply(lambda x: x.days)
    active = clean_df.FirstSubmission.notnull()
    user_activity = ['list of user activity features']
    user_activity_des = pd.DataFrame(clean_df.loc[active][user_activity].describe())
    user_activity_des.to_csv(r"Data\user_activity_des.csv")
    return user_activity_des

def cleanData_activity(df):
    '''
    Clean the following variables in the activity data.
    Time stamp variables: keep date only.
    '''
    timeFeature = ['list of features that are timestamps']
    for col in timeFeature:
        df[col] = pd.to_datetime(df[col], errors ='coerce').dt.date
        newTime = replaceString(col, ['Timestamp','Utc'], '')
        df = df.rename(columns={col:newTime})

    dollarFeature = df.columns[df.columns.str.endswith('InCents')].tolist()
    for col in dollarFeature:
        df[col] = np.round(df[col]/100000, 2)
        newDollar = replaceString(col, ['InCents',], 'T')
        df = df.rename(columns={col:newDollar})

    variables = df.columns.tolist()
    return df, variables

def cleanData_deal(df):
    '''
    Clean the deal data.
    Time stamp variables: keep date only.
    '''
    df['equity_ratio'] = np.round(df['equity']/df['capital']*100, 2)

    timeFeature = ['list of features that are timestamps']
    for col in timeFeature:
        df[col] = pd.to_datetime(df[col], errors ='coerce').dt.date
        newTime = replaceString(col, ['Timestamp',], '')
        df = df.rename(columns={col:newTime})

    dollarFeature = df.columns[df.columns.str.endswith('InCents')].tolist()
    for col in dollarFeature:
        df[col] = np.round(df[col]/100000, 2)
        newDollar = replaceString(col, ['activitytedAmountInCents','activityAmountInCents'], 'CAT')
        df = df.rename(columns={col:newDollar})

    df = df.drop(['list of redundent features'],axis=1)

    capitalFeature = list(df.columns[df.columns.str.startswith('Capital')])
    df = df.drop(capitalFeature, axis=1)
    mask = (df.dealstart>dt.date(1971,1,1))
    df = df.loc[mask]
    variables = list(df.columns)
    return df, variables

def cleanData_additionaldeal(df):
    '''
    Clean the additional deal data with IRR info.
    '''
    mask = ['list of features whose units need to be changed',]
    for col in mask:
        df[col] = np.round(df[col]/1000,2)
        newReturn = replaceString(col, ['Projected','InThousandths'], '')
        df = df.rename(columns={col:newReturn})
    df = df.fillna(0)
    df['multiple'] = np.round((df['multiple_high'] + df['multiple_loww'])/2,2)
    df['irr'] = np.round((df['irr_high'] + df['irr_low'])/2,2)
    df['return'] = np.round((df['ret_high'] + df['ret_low'])/2, 2)
    df = df[['dealId','multiple', 'irr', 'return']]
    return df

def merge_deal(df1, df2):
    '''
    merge clean_deal, clean_add_deal together
    '''
    df3 = df1.merge(df2, how='left', on='dealId')
    return df3

def cleanData_manager(df):
    '''
    Clean the manager data
    '''
    useless_column = ['list of useless features']
    df = df.drop(useless_column, axis=1)
    dollarFeature = df.columns[df.columns.str.endswith('InCents')].tolist()
    for col in dollarFeature:
        df[col] = np.round(df[col]/100000, 2)
        newDollar = replaceString(col, ['InCents'], 'T')
        df = df.rename(columns={col:newDollar})
    
    return_column = ['list of features that describe returns']
    for col in return_column:
        df[col] = np.round(df[col]/10,2)
        newReturn = replaceString(col, ['InTenths'], '')
        df = df.rename(columns={col:newReturn})    
    
    assetFeature = df.columns[df.columns.str.startswith('management')].tolist()
    for col in assetFeature:
        newAsset = replaceString(col, ['management'], 'AUM')
        df = df.rename(columns={col:newAsset})
    df['multiple_in_pct'] = np.round(df['multiple_in_hundth']/100, 2)
    df = df.rename(columns={"multiple_in_pct":"avg_multiple"})
    df = df.fillna(0)
    variables = df.columns.tolist()
    return df, variables

def cross_join(deal, user):
    '''
    Create cross pairing, i.e. cartesian product between deal and user data
    that meets certain conditions:
        1. UserID must not be null.
        2. User must create her account before a deal ends, otherwise there is no chance
        for her to act.
        3. Deal priori is the set of information that is available to users when they
        see this deal online.
        4. User_useful are the list of variables that potentially could be useful in determining
        whether they are going to act or not.
        5. Determine the order among user activity and deal end date. 
        If any of the user activities happens after a deal closes, it is not used to 
        predict whether an user will act in a deal or not.
        All variables like LifetimeAverageDaysBetween (LABD) are therefore not
        appropriate for a prediction analysis. Focus on running average,
        i.e. the average of user behavior up to a certain point in time.
    '''
    deal['key'] = 1
    deal_priori = ['list of features that are available at the time of a deal start']
    deal = deal[deal_priori]
    user['key'] = 1
    active_user = (user.zipcode.notnull())
    user = user.loc[active_user]
    user_useful =['list of usefull user features']
    user = user[user_useful]
    pairing = pd.merge(deal, user, on='key')
    print("there are " + str(pairing.shape[0]) + " potential pairings between", end=' ')
    print("deals and potentially active users.")
    mask1 = (pairing.UserId.notnull())
    mask2 = (pairing.deal_end > pairing.originDate)
    pairing = pairing.loc[mask1 & mask2]
    print("there are " + str(pairing.shape[0]) + " feasible pairings between", end=' ')
    print("deals and potentially active users.")
    pairing = pairing.drop_duplicates().drop(['key',], axis=1)

    firstactivity = pairing['FirstActivity'].tolist()
    sndactivity = pairing['SecondActivity'].tolist()
    lastactivity = pairing['LastActivity'].tolist()
    dealend = pairing['deal_end'].tolist()

    for i in range(0, len(firstactivity)):
        if pd.notna(firstactivity[i]):
            if firstactivity[i] > dealend[i]:
                firstactivity[i] = np.nan
    for i in range(0, len(sndactivity)):
        if pd.notna(sndactivity[i]):
            if sndactivity[i] > dealend[i]:
                sndactivity[i] = np.nan
    for i in range(0, len(lastactivity)):
        if pd.notna(lastactivity[i]):
            if lastactivity[i] > dealend[i]:
                lastactivity[i] = np.nan

    pairing['FirstActivity'] = pd.Series(firstactivity)
    pairing['SecondActivity'] = pd.Series(sndactivity)
    pairing['LastActivity'] = pd.Series(lastactivity)
    pairing['activity'] = pairing[['FirstActivity', 'SecondActivity', 'LastActivity']].values.tolist()

    no_nan_set = lambda s: {x for x in s if x == x}
    pairing['activity_exp'] = pairing['activity'].apply(lambda x:len(no_nan_set(x)))
    pairing = pairing.drop(['activity','FirstActivity', 'SecondActivity', 'LastActivity'], axis=1)    
        
    hpi_3zip= pd.read_excel(r"Data\HPI_AT_3zip.xlsx",dtype={'Three-Digit ZIP Code': str})
    hpi_3zip = hpi_3zip.rename(columns={"Three-Digit ZIP Code":"thr_zip", "Index (NSA)":"hpi"})
    hpi_3zip = hpi_3zip.drop(['Index Type'],axis=1)
    hpi_3zip['qtr_ret'] = hpi_3zip.groupby(['thr_zip'])['hpi'].pct_change()
    hpi_3zip = hpi_3zip.dropna()
    
    pairing['thr_zip'] = pairing['zipcode'].str.slice(0,3)
    pairing['deal_start'] = pd.to_datetime(pairing['deal_start'],errors='coerce')
    pairing['Year'] = pairing['deal_start'].dt.year
    pairing['Quarter'] = pairing['deal_start'].dt.quarter
    pairing['deal_start'] = pairing['deal_start'].dt.date
    pairing = pairing.merge(hpi_3zip, how='left', on=['thr_zip','Year','Quarter'])
    pairing['deal_region_qtr_ret'] = np.round(pairing['qtr_ret'] * 100, 2)

    pairing = pairing.reset_index(drop=True)
    return pairing

def join_master(pairing, activity, manager):
    '''
    Merge all possible pairings between deal and users with the actual "activity"
    made by the user at one point in time:
        1. The outcome will be binary: 0 for no activity in a deal,
        and 1 for activity in a deal.
    '''
    activity_useful = ['list of useful features in activity']

    user_waitlist = activity[['UserId','WaitList']].drop_duplicates()
    mask1 = (user_waitlist['WaitList']>dt.date(2010,1,1))
    user_waitlist = pd.DataFrame(user_waitlist[mask1].groupby(['UserId'])['WaitList'].min()).reset_index()
    user_waitlist = user_waitlist.rename(columns={"WaitList":"fst_waitlist"})

    master = pairing.merge(activity[activity_useful], how='left',
                           left_on=['dealId','UserId',], 
                           right_on=['dealId','UserId'])
    master = master.merge(user_waitlist, how='left', on='UserId')

    master_copy = master.copy()

    running_activity = activity[['UserId','Submit','RunningTotalT']].drop_duplicates()
    master = master.merge(running_activity, how='left', on='UserId')

    running_total = master[['list of features related to running total']].drop_duplicates()

    running_total['days_diff'] = (running_total['Submit']- 
                 running_total['deal_start']).apply(lambda x:x.days)
    running_total = running_total[running_total['days_diff']<0]

    max_days = pd.DataFrame(running_total.groupby(['dealId','UserId'])['days_diff'].max()).reset_index()
    max_days = max_days.drop_duplicates()

    running_total = running_total.merge(max_days, how='inner', on=['dealId','UserId','days_diff'])
    running_total = running_total.rename(columns={"RunningTotalT":"prior_input"})
    running_total = running_total.groupby(['deakId','UserId']).tail(1)

    master_copy = master_copy.merge(running_total[['dealId','UserId', 'prior_input', 'days_diff']],
                                    how='left', on=['dealId','UserId'])
    master_copy['days_diff'] = master_copy['days_diff'] * (-1)
    master_copy = master_copy.rename(columns={"days_diff":"prior_activity"})
    master_copy['prior_input'] = master_copy['prior_input'].fillna(0)
    del master
    master_copy = master_copy.drop_duplicates()
    master_copy = master_copy.merge(manager, how='left',on=['managerId'])
    
    master_copy['interested'] = 0
    master_copy.loc[master_copy['dealId'].notnull(), 'interested'] = 1

    master_copy['acted'] = 0
    acted = (master_copy['Status']==2) & ((master_copy['actComplete'].notnull())|(master_copy['paperComplete'].notnull()))
    master_copy.loc[acted, 'acted'] = 1

    master_copy['ever_waitlisted'] = 0
    master_copy.loc[master_copy['fst_waitlist'] < master_copy['deal_start'], 'ever_waitlisted'] = 1

    deal_region = master_copy['Location'].tolist()
    for i in range(0, len(deal_region)):
        try:
            deal_region[i] = deal_region[i].split(',')[1].lstrip()
        except (ValueError, IndexError, AttributeError):
            deal_region[i]= deal_region[i]
    master_copy['deal_region'] = pd.Series(deal_region)

    master_copy['home_act'] = 0
    master_copy.loc[master_copy['deal_region']==master_copy['new_region'], 'home_act'] = 1
    
    median_home_value = []
    median_household_income = []
    search = SearchEngine(simple_zipcode=True)
    print("Start searching for user zipcode's median home value and median household income....")
    for ele in master_copy['zipcode']:
        ele_search = search.by_zipcode(ele)
        median_home_value.append(ele_search.median_home_value)
        median_household_income.append(ele_search.median_household_income)
    master_copy['user_median_home_value'] = pd.Series(median_home_value)
    master_copy['user_median_household_income'] = pd.Series(median_household_income)

    master_copy['AmountT'] = master_copy['AmountT'].fillna(0)
    print("Saving master data into file.....")
    master_copy.to_csv(r"Data\master.csv")
    print("Master file saved....")
    return master_copy

def generate_master_data():
    dataTypes, manager, deal, activity, users = loadData(r"Data\anonymized-data.xlsx")
    add_deal = loadData_additionaldeal(r"Data\additional-deal-data.csv")

    clean_user, user_variables = cleanData_user(users)
    user_activity_des = user_Analysis(clean_user)
    clean_activity, activity_variables = cleanData_activity(activity)
    clean_deal, deal_variables = cleanData_deal(deal)
    clean_manager, manager_variables = cleanData_manager(manager)
    clean_add_deal = cleanData_additionaldeal(add_deal)

    deal = merge_deal(clean_deal, clean_add_deal)
    pairing = cross_join(deal, clean_user)
    #make sure the pairing dataframe now has the total_view_so_far variable
    master = join_master(pairing, clean_activity, clean_manager)
    
    del US_state, US_state1, US_state2
    del activity, deal, users, manager

    return master


