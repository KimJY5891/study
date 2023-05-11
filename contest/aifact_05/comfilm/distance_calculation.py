
#마지막에 3으로 바꾸면 3개 나온다. 
import math
import numpy as np
import pandas as pd

path_meta = './_data/aifact_05/META/'

# read in the csv files
pmmap_csv = pd.read_csv(path_meta+'pmmap.csv', index_col=False, encoding='utf-8')
awsmap_csv = pd.read_csv(path_meta+'awsmap.csv', index_col=False, encoding='utf-8')

# create a dictionary of places in awsmap.csv
places_awsmap = {}
for i, row in awsmap_csv.iterrows():
    places_awsmap[row['Description']] = (row['Latitude'], row['Longitude'])

# define a function to calculate the distance between two points
def distance(lat1, lon1, lat2, lon2):
    R = 6371 # earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# loop over the locations in pmmap.csv
for i, row_a in pmmap_csv.iterrows():
    # find the closest places in awsmap.csv to the location in pmmap.csv
    closest_places = []
    closest_distances = []
    for j, row_b in awsmap_csv.iterrows():
        dist = distance(row_a['Latitude'], row_a['Longitude'], row_b['Latitude'], row_b['Longitude'])
        if len(closest_places) < 3:
            closest_places.append(row_b['Location'])
            closest_distances.append(dist)
        else:
            max_index = closest_distances.index(max(closest_distances))
            if dist < closest_distances[max_index]:
                closest_places[max_index] = row_b['Location']
                closest_distances[max_index] = dist
    
    # sort the distances in ascending order
    closest_places = [x for _, x in sorted(zip(closest_distances, closest_places))]
    closest_distances.sort()
    
    # print the closest places for the location in pmmap.csv with their distances
    print("Closest places to {} (in ascending order of distance):".format(row_a['Location']))
    for k in range(1):
        print("{}, distance: {:.2f} km".format(closest_places[k], closest_distances[k]))
