# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 22:28:41 2019

@author: diana
This document plots the user physical location.
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from uszipcode import SearchEngine
from uszipcode import Zipcode

def user_location(clean_user):
    search = SearchEngine(simple_zipcode=True)
    
    inv_totalcat = pd.DataFrame(clean_user.groupby(['zipcode'])['TotalCAT'].sum()).reset_index().drop_duplicates()
    inv_totalcmt = pd.DataFrame(clean_user.groupby(['zipcode'])['Totalactivitys'].sum()).reset_index().drop_duplicates()
    inv_total = inv_totalcat.merge(inv_totalcmt, how='left', on=['zipcode'])
    zipcode = inv_total[inv_total['TotalCAT']>0].reset_index(drop=True)
    
    inv_lat = []
    inv_lon = []
    for ele in zipcode['zipcode']:
        ele_search = search.by_zipcode(ele)
        inv_lat.append(ele_search.lat)
        inv_lon.append(ele_search.lng)
        
    zipcode['inv_lat'] = pd.Series(inv_lat)
    zipcode['inv_lon'] = pd.Series(inv_lon)   
    zipcode = zipcode.dropna().reset_index(drop=True) 
        
    lat = zipcode['inv_lat'].values
    lon = zipcode['inv_lon'].values
    total_activity = zipcode['Totalactivitys'].values
    total_CAT = zipcode['TotalCAT'].values
    
    # 1. Draw the map background
    fig = plt.figure(figsize=(30, 30))
    m = Basemap(projection='lcc', resolution='h', 
                lat_0=37.09, lon_0=-95.71,
                width=6E6, height=4E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')
    
    # 2. scatter city data, with color reflecting population
    # and size reflecting area
    m.scatter(lon, lat, latlon=True,
              s=total_CAT, c=total_activity,
              cmap='Reds', alpha=1)
    
    # 3. create colorbar and legend
    cbar = plt.colorbar(label=r'Number of Total Activity')
    cbar.ax.tick_params(labelsize=20) 
    #plt.clim(30, 70)
    
    # make legend with dummy points
    for a in [100, 200, 400]:
        plt.scatter([], [], c='k', alpha=1, s=a,
                    label=str(a) + ' Thousand $ in Total activity Amount')
    plt.legend(title='Size of the dot indicates...', scatterpoints=1, frameon=False,
               labelspacing=1, loc='lower left')
    plt.title('user Location in the USA',fontsize=20)
    plt.tight_layout()
    plt.savefig('user location in the USA.png')
    plt.show()
    return zipcode

def deal_location(deal):
    offer_location = deal['Location'].tolist()
    search = SearchEngine(simple_zipcode=True)
    
    city_list = []
    state_list = []
    zipcode_list = []
    for location in offer_location:
        try:
            city = location.split(',')[0].lstrip().lower().replace('msa','')
            state = location.split(',')[1].lstrip().lower().replace('msa','')
            zipcode = search.by_city_and_state(city, state, sort_by=Zipcode.population, 
                                           ascending=False, returns=1)[0].zipcode
        except (ValueError, IndexError, AttributeError):
            city = 'nan'
            state = 'nan'
            zipcode = 'nan'
        city_list.append(city)
        state_list.append(state)
        zipcode_list.append(zipcode)
        
    deal['zipcode'] = pd.Series(zipcode_list)   
    deal = deal[deal['zipcode'] != 'nan']
    deal_totalcat = pd.DataFrame(deal.groupby(['zipcode'])['TotalCAT'].sum()).reset_index().drop_duplicates()
    deal_totalcmt = pd.DataFrame(deal.groupby(['zipcode'])['Totalactivitys'].sum()).reset_index().drop_duplicates()
    off_total = deal_totalcat.merge(deal_totalcmt, how='left', on=['zipcode'])
    off_zipcode = off_total[off_total['TotalCAT']>0].reset_index(drop=True)
    
    off_lat = []
    off_lon = []
    for ele in off_zipcode['zipcode']:
        ele_search = search.by_zipcode(ele)
        off_lat.append(ele_search.lat)
        off_lon.append(ele_search.lng)
        
    off_zipcode['deal_lat'] = pd.Series(off_lat)
    off_zipcode['deal_lon'] = pd.Series(off_lon)   
    off_zipcode = off_zipcode.dropna().reset_index(drop=True) 
        
    lat = off_zipcode['deal_lat'].values
    lon = off_zipcode['deal_lon'].values
    total_activity = off_zipcode['Totalactivitys'].values
    total_CAT = off_zipcode['TotalCAT'].values
    
    # 1. Draw the map background
    fig = plt.figure(figsize=(30, 30))
    m = Basemap(projection='lcc', resolution='h', 
                lat_0=37.09, lon_0=-95.71,
                width=6E6, height=4E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')
    
    # 2. scatter city data, with color reflecting population
    # and size reflecting area
    m.scatter(lon, lat, latlon=True,
              s=total_CAT, c=total_activity,
              cmap='Blues', alpha=1)
    
    # 3. create colorbar and legend
    cbar = plt.colorbar(label=r'Number of Total Activity in deal')
    cbar.ax.tick_params(labelsize=20) 
    #plt.clim(30, 70)
    
    # make legend with dummy points
    for a in [400, 1000, 3000]:
        plt.scatter([], [], c='blue', alpha=1, s=a,
                    label=str(a) + ' Thousand $ in Total activity Amount in deals')
    plt.legend(title='Size of the dot indicates...', scatterpoints=1, frameon=False,
               labelspacing=1, loc='lower left')
    plt.title('deal Location in the USA',fontsize=20)
    plt.tight_layout()
    plt.savefig('deal location in the USA.png')
    plt.show()
    return off_zipcode

