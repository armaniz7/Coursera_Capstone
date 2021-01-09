#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://ibm.box.com/shared/static/9gegpsmnsoo25ikkbl4qzlvlyjbgxs5x.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Segmenting and Clustering Neighborhoods in Toronto</font></h1>
# 

# Before we get the data and start exploring it, let's download all the dependencies that we will need.
# 

# # Part 1

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# <a id='item1'></a>
# 

# ## 1. Download and Explore Dataset
# 

# In[70]:


#downloaded the files and scrapped the data out wikipedia page 
from IPython.display import display_html
source = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup=BeautifulSoup(source,'lxml')
print(soup.title)
tab = str(soup.table)
display_html(tab, raw=True)


# Create dataframe from our data 

# In[64]:


dfs = pd.read_html(tab)
df=dfs[0]
df.head()


# In[72]:


# dropping cells with a borough that is Not assigned.
df1 = df[df.Borough != 'Not assigned']

# Combining the neighbourhoods that shares the same Postalcode
toronto_df = df1.groupby(['Postal Code','Borough'], sort=False).agg(', '.join)
toronto_df.reset_index(inplace=True)


#Making the neighborhood the same as the borough, a cell has a borough but a Not assigned 
toronto_df['Neighbourhood'] = np.where(df2['Neighbourhood'] == 'Not assigned',df2['Borough'], df2['Neighbourhood'])


# In[73]:


toronto_df.head()


# In[74]:


# shape method to print the number of rows of your dataframe.
toronto_df.shape


# # Part 2

# Using the Geocoder package or the csv file to add 'Latitude' and 'Longitutde' to the toronto_df

# In[75]:


lat_lon = pd.read_csv('https://cocl.us/Geospatial_data')
lat_lon.head()


# In[76]:


toronto_df_new = pd.merge(toronto_df,lat_lon,on='Postal Code')
toronto_df_new.head()


# # Part 3 

# #### Load and explore the data
# 

# Next, let's load the data.
# 

# Let's take a quick look at the data.
# 

# And make sure that the dataset has all 10 boroughs and 103 neighborhoods.
# 

# In[80]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(toronto_df_new['Borough'].unique()),
        toronto_df_new.shape[0]
    )
)


# #### Use geopy library to get the latitude and longitude values of Toronto
# 

# In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent <em>Toronto Explorer</em>, as shown below.
# 

# In[81]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="Toronto Explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# #### Create a map of Toronto Explorer with neighborhoods superimposed on top.
# 

# In[84]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_df_new['Latitude'], toronto_df_new['Longitude'], toronto_df_new['Borough'], toronto_df_new['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[87]:


Downtown_Toronto_data = toronto_df_new[toronto_df_new['Borough'] == 'Downtown Toronto'].reset_index(drop=True)
Downtown_Toronto_data.head()


# Let's get the geographical coordinates of Downtown Toronto.
# 

# In[ ]:


address = 'Downtown Toronto, ON'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Downtown Toronto are {}, {}.'.format(latitude, longitude))


# As we did with all of Toronto, let's visualizat Downtown Toronto the neighborhoods in it

# In[88]:


# create map of Toronto using latitude and longitude values
map_Downtown_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(Downtown_Toronto_data['Latitude'], Downtown_Toronto_data['Longitude'], Downtown_Toronto_data['Neighbourhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Downtown_Toronto)  
    
map_Downtown_Toronto


# Next, we are going to start utilizing the Foursquare API to explore the neighborhoods and segment them.
# 

# #### Define Foursquare Credentials and Version
# 

# In[89]:


CLIENT_ID = 'GSBPKOLYKWIXWODEAEDMKMO0H0DHPRUQ1RLLKXT4OFGS0HJG' 
CLIENT_SECRET = 'PILEGZLMDCJW3RD4CBK1CAWYRWQFVAADYHAMI3QMDHENWRBD'
VERSION = '20180605' 
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Let's explore the first neighborhood in our dataframe.
# 

# Get the neighborhood's name.
# 

# In[91]:


Downtown_Toronto_data.loc[0, 'Neighbourhood']


# Get the neighborhood's latitude and longitude values.
# 

# In[93]:


neighborhood_latitude =Downtown_Toronto_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude =Downtown_Toronto_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = Downtown_Toronto_data.loc[0, 'Neighbourhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# #### Now, let's get the top 100 venues that are in Regent Park within a radius of 500 meters.
# 

# First, let's create the GET request URL
# 

# In[94]:


# type your answer here

LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url 


# Send the GET request and examine the resutls
# 

# In[95]:


results = requests.get(url).json()
results


# From the Foursquare lab in the previous module, we know that all the information is in the _items_ key. Before we proceed, let's borrow the **get_category_type** function from the Foursquare lab.
# 

# In[96]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# Now we are ready to clean the json and structure it into a _pandas_ dataframe.
# 

# In[97]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# And how many venues were returned by Foursquare?
# 

# In[98]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# <a id='item2'></a>
# 

# ## 2. Explore Neighborhoods in Downtown Toronto
# 

# #### Let's create a function to repeat the same process to all the neighborhoods in Toronto
# 

# In[99]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# #### Now write the code to run the above function on each neighborhood and create a new dataframe called Downtown_Toronto_venues
# 

# In[102]:



Downtown_Toronto_venues = getNearbyVenues(names=Downtown_Toronto_data['Neighbourhood'],
                                   latitudes=Downtown_Toronto_data['Latitude'],
                                   longitudes=Downtown_Toronto_data['Longitude']
                                  )


# #### Let's check the size of the resulting dataframe
# 

# In[103]:


print(Downtown_Toronto_venues.shape)
Downtown_Toronto_venues.head()


# Let's check how many venues were returned for each neighborhood
# 

# In[106]:


Downtown_Toronto_venues.groupby('Neighborhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues
# 

# In[107]:


print('There are {} uniques categories.'.format(len(Downtown_Toronto_venues['Venue Category'].unique())))


# <a id='item3'></a>
# 

# ## 3. Analyze Each Neighborhood
# 

# In[109]:


# one hot encoding
Downtown_Toronto_onehot = pd.get_dummies(Downtown_Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Downtown_Toronto_onehot['Neighborhood'] = Downtown_Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [Downtown_Toronto_onehot.columns[-1]] + list(Downtown_Toronto_onehot.columns[:-1])
Downtown_Toronto_onehot = Downtown_Toronto_onehot[fixed_columns]

Downtown_Toronto_onehot.head()


# And let's examine the new dataframe size.
# 

# In[110]:


Downtown_Toronto_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
# 

# In[111]:


Downtown_Toronto_grouped = Downtown_Toronto_onehot.groupby('Neighborhood').mean().reset_index()
Downtown_Toronto_grouped


# #### Let's confirm the new size
# 

# In[113]:


Downtown_Toronto_grouped.shape


# There are 19 neigbourhoods with 207 categories 

# #### Let's print each neighborhood along with the top 5 most common venues
# 

# In[114]:


num_top_venues = 5

for hood in Downtown_Toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Downtown_Toronto_grouped[Downtown_Toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a _pandas_ dataframe
# 

# First, let's write a function to sort the venues in descending order.
# 

# In[115]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.
# 

# In[118]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Downtown_Toronto_grouped['Neighborhood']

for ind in np.arange(Downtown_Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Downtown_Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[152]:


neighborhoods_venues_sorted


# In[149]:


neighborhoods_venues_sorted['1st Most Common Venue'].value_counts()


# In[150]:


neighborhoods_venues_sorted['2nd Most Common Venue'].value_counts()


# In[151]:


neighborhoods_venues_sorted['3rd Most Common Venue'].value_counts()


# ### Coffee Shop is the most common venue in Neigborhood, which is not surprising considering that we analyzed Downtown Toronto.
# 

# ## 4. Cluster Neighborhoods
# 

# Run _k_-means to cluster the neighborhood into 5 clusters.
# 

# In[120]:


# set number of clusters
kclusters = 5

Downtown_Toronto_grouped_clustering = Downtown_Toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Downtown_Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
# 

# In[138]:



Downtown_Toronto_merged = Downtown_Toronto_data

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
Downtown_Toronto_merged= Downtown_Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')
Downtown_Toronto_merged # check the last columns!
Downtown_Toronto_merged.tail()


# Finally, let's visualize the resulting clusters
# 

# In[140]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Downtown_Toronto_merged['Latitude'], Downtown_Toronto_merged['Longitude'], Downtown_Toronto_merged['Neighbourhood'], Downtown_Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### There are more frequent intervals within Clusters located closer to the harbour than other clusters

# <a id='item5'></a>
# 

# ## 5. Examine Clusters
# 

# 
# 

# #### Cluster 1
# 

# In[142]:


Downtown_Toronto_merged.loc[Downtown_Toronto_merged['Cluster Labels'] == 0, Downtown_Toronto_merged.columns[[1] + list(range(5, Downtown_Toronto_merged.shape[1]))]]


# Coffe Shops are the most common value following by restaurants, hotels, bars. The result is normal because downtown Toronto might be the most visited place both for tourists and locals.

# #### Cluster 2
# 

# In[144]:


Downtown_Toronto_merged.loc[Downtown_Toronto_merged['Cluster Labels'] == 1, Downtown_Toronto_merged.columns[[1] + list(range(5, Downtown_Toronto_merged.shape[1]))]]


# Cluster 2 has more social areas like parks, playgrounds and the transportation is good and can be done by rail. People might find this area as a good place to live in.

# #### Cluster 3
# 

# In[143]:


Downtown_Toronto_merged.loc[Downtown_Toronto_merged['Cluster Labels'] == 2, Downtown_Toronto_merged.columns[[1] + list(range(5, Downtown_Toronto_merged.shape[1]))]]


# #### Cluster 4
# 

# In[145]:


Downtown_Toronto_merged.loc[Downtown_Toronto_merged['Cluster Labels'] == 3, Downtown_Toronto_merged.columns[[1] + list(range(5, Downtown_Toronto_merged.shape[1]))]]


# Cluster 4 is the airport area

# #### Cluster 5
# 

# In[146]:


Downtown_Toronto_merged.loc[Downtown_Toronto_merged['Cluster Labels'] ==4, Downtown_Toronto_merged.columns[[1] + list(range(5, Downtown_Toronto_merged.shape[1]))]]


# Cluster 5 has coffe shops as the most commmon value behind the cluster 1 which has more coffe shops than cluster 5. Cluster 1 is less popular than Cluster 5. It might be the area where locals go outside and spend their lesiure times
