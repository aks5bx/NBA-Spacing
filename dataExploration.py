#%%
##############################
### IMPORT LIBRARIES, DATA ###
##############################

import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
tqdm_notebook().pandas()

def x_round(x):
    return math.floor(x*4)/4

## ptd - player tracking data
ptd = pd.read_csv('data/csv/0021500490.csv')
ptd['game_clock'] = ptd.apply(lambda row: x_round(row['game_clock']), axis=1)
ptd_shortened = ptd.drop_duplicates(subset= ['game_clock', 'quarter', 'player_id'], keep='last')

## pbp - play by play
pbp = pd.read_csv('data/events/0021500490.csv')
# pbp = pbp[['EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 'PERIOD',
#        'PCTIMESTRING', 'HOMEDESCRIPTION',
#        'VISITORDESCRIPTION', 'SCORE', 'SCOREMARGIN']]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


#%%
############################################
### CONVERT DATA TYPES, JOIN DATA FRAMES ###
############################################

## convert the datetime formats
def convertTime(time_val):
    try:
        val = (int(time_val.split(':')[0]) * 60) + (int(time_val.split(':')[1]))
    except:
        print(time_val)
    
    return val

## apply time conversion & merge two data frames
pbp['game_clock'] = pbp.apply(lambda row: convertTime(row['PCTIMESTRING']), axis=1)
full_game_data = ptd_shortened.merge(pbp, left_on = ['game_clock', 'quarter'], right_on = ['game_clock', 'PERIOD'], how = 'left')

## fill na's with previous valid value, add points scored 
full_game_data['SCORE'] = full_game_data['SCORE'].fillna(method='ffill')
full_game_data['SCOREMARGIN'] = full_game_data['SCOREMARGIN'].fillna(method='ffill')
full_game_data.loc[full_game_data.SCOREMARGIN == 'TIE', 'SCOREMARGIN'] = 0
full_game_data['SCOREMARGIN'] = full_game_data['SCOREMARGIN'].fillna(0)
full_game_data['SCOREMARGIN'] = full_game_data['SCOREMARGIN'].astype('int')
full_game_dataStripped = full_game_data.drop_duplicates(subset= ['game_clock', 'quarter', 'player_id'], keep='last')


## Note: full_game_data Primary Key is: game_block + shot_clock + quarter

#%%
###########################
### ADD RELEVANT FIELDS ###
###########################

## did someone attempt a shot 
full_game_data['shot_taken'] = ((full_game_data.EVENTMSGTYPE == 1) | (full_game_data.EVENTMSGTYPE == 2)).astype('int')

## did someone make a shot
full_game_data['shot_made'] = ((full_game_data.EVENTMSGTYPE == 1).astype('int'))

## which team scored
full_game_data['team_scored'] = np.where(full_game_data['shot_made'] == 1, full_game_data['PLAYER1_TEAM_ID'],'')

## did the offensive player get fouled 
## event msg action type of 26 is an offensive foul
full_game_data['foul_reg'] = ((full_game_data.EVENTMSGTYPE == 6.0) & (full_game_data.EVENTMSGACTIONTYPE != 26.0)).astype('int')

## TO DO: define who got fouled


##############################
### DEFINE TEAMS, SPACING  ###
##############################

teamList = list(set(full_game_data.team_id))
teamList.remove(-1)
teamDict = dict(zip(['team1', 'team2'], teamList))

full_game_data['team1_hullSpacing'] = 0
full_game_data['team2_hullSpacing'] = 0


# %%
###################################
## SPACING METHOD 1: CONVEX HULL ##
###################################

### Some notes on the size of a basketball court
## 29 feet top of the arc to baseline 
## 41 from outer circle to baseline 
## 35 for spacing calculations 

## Function takes in a dataframe and returns convex hull spacing 
def hullSpacing(playDF):
    points = playDF[['x_loc','y_loc']].to_numpy()
    hull = ConvexHull(points)
    return hull.volume


## Function defines a dataframe based on row 
def computeHullSpacing(row):
    try:
        if teamDict['team1'] == row['team_id']:
            team = 'team1'
        elif teamDict['team2'] == row['team_id']:
            team = 'team2'
        else:
            return -99

        if team == 'team1':
            if row['team1_hullSpacing'] > 0:
                pass
            else:
                subset_df = full_game_data[(full_game_data.team_id == row['team_id']) & (full_game_data.game_clock == row['game_clock']) & (full_game_data.quarter == row['quarter'])] 
                
                hullSpacingVal = hullSpacing(subset_df)

                full_game_data.loc[ ( (full_game_data['game_clock'] == row['game_clock'])  &  (full_game_data['quarter'] == row['quarter'])  ) , 'team1_hullSpacing'] = hullSpacingVal

        elif team == 'team2':
            if row['team2_hullSpacing'] > 0:
                pass
            else:
                subset_df = full_game_data[(full_game_data.team_id == row['team_id']) & (full_game_data.game_clock == row['game_clock']) & (full_game_data.quarter == row['quarter'])] 
                
                hullSpacingVal = hullSpacing(subset_df)

                full_game_data.loc[ ( (full_game_data['game_clock'] == row['game_clock']) & (full_game_data['quarter'] == row['quarter'])  ) , 'team2_hullSpacing'] = hullSpacingVal
    
    except:
        print('---------------------------------')
        print(row)


full_game_data.progress_apply(lambda row: computeHullSpacing(row), axis=1)



#%%
#################################
### SPACING IMPACT ON SCORING ###
#################################

## Spacing vs Scoring Rate
def generateSpacingScoringTable(team):
    ssDict = {}
    ind = 0
    thresholds = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 100000]

    if team == 'team1':

        for threshold in thresholds:
            teamSpacingThreshold = full_game_data[(full_game_data.team1_hullSpacing > threshold) & (full_game_data.team1_hullSpacing <= thresholds[ind + 1])]

            teamScoringRate_total = len(teamSpacingThreshold[teamSpacingThreshold.team_scored == str(float(teamDict[team]))]) / len(teamSpacingThreshold)

            ssDict[threshold] = teamScoringRate_total
            ind += 1

            if thresholds[ind] > 1000:
                break

        resDF = pd.DataFrame(ssDict.items(), columns = ['Spacing Range', 'Scoring Rate'])

        resDF.plot.scatter(x='Spacing Range', y='Scoring Rate')

        return resDF

    if team == 'team2':

        for threshold in thresholds:
            teamSpacingThreshold = full_game_data[(full_game_data.team2_hullSpacing > threshold) & (full_game_data.team2_hullSpacing <= thresholds[ind + 1])]

            teamScoringRate_total = len(teamSpacingThreshold[teamSpacingThreshold.team_scored == str(float(teamDict[team]))]) / len(teamSpacingThreshold)

            ssDict[threshold] = teamScoringRate_total
            ind += 1

            if thresholds[ind] > 1000:
                break

        resDF = pd.DataFrame(ssDict.items(), columns = ['Spacing Range', 'Scoring Rate'])

        resDF.plot.scatter(x='Spacing Range', y='Scoring Rate')

        return resDF

## Plot points 
# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


# %%
########################################
### SPACING METHOD 2: UNGUARDED AREA ###
########################################

# 1. Calculate total area of the polygon 
# 2. calculate total area of the area around the point (25 pi), let’s call this A_Point
# 3. Find the intersection between the area of the shape and A_Point
# 4. Use the value in step 3 to determine the amount of A_Point that exists outside of the polygon, call that A_Point2
# 5. Do total area of polygon - A_Point2
from shapely.geometry.polygon import LinearRing, Polygon
import math
pi = math.pi

sampleFGD = full_game_data[(full_game_data.game_clock == 718.50) & (full_game_data.quarter == 1) & (full_game_data.team_id != -1)]

courtPolygon = Polygon([(0,0),(0,50),(94,50), (94,0)])

## Takes a dataframe and returns a list of polygons
def polygonList(df, radius):
    polygonList = []
    for idx, row in df.iterrows():
        xCoor = row['x_loc']
        yCoor = row['y_loc']
        pts = generateCircle(xCoor, yCoor, radius, 20)
        poly = Polygon(pts).buffer(0)

        polygonList.append(poly)

    return polygonList

## Takes an x-coordinates, y-coordinate, radius, number of points, and generates a circle 
def generateCircle(xCoor, yCoor, radius, n):
    return [((math.cos(2*pi/n*x)* radius + xCoor) , (math.sin(2*pi/n*x)* radius) + yCoor) for x in range(0,n+1)]

## Takes in points and plots the circle
def plotCircle(points):
    ring = LinearRing(points)
    x, y = ring.xy
    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color='#6699cc', alpha=0.7,linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_title('Polygon')

## Returns the intersection area of two polgygons
def intersectionArea(p1, p2):
    return p1.intersection(p2).area

## Generates the area of a circle
def generatePointArea(r):
    return pi * (r * r)

## takes points, a list of polygons, returns the intersection of polygons and a court
def courtIntersection(points, r):
    circleArea = generatePointArea(r)

    totalIntersection = 0
    for point in points: 
        totalIntersection += (circleArea - intersectionArea(point, courtPolygon))

    return totalIntersection

## takes points, a list of polygons, returns total intersecting area
def pointsIntersection(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]

    total = 0
    total += intersectionArea(p1, p2)
    total += intersectionArea(p1, p3)
    total += intersectionArea(p1, p4)
    total += intersectionArea(p1, p5)
    total += intersectionArea(p2, p3)
    total += intersectionArea(p2, p4)
    total += intersectionArea(p2, p5)
    total += intersectionArea(p3, p4)
    total += intersectionArea(p3, p5)
    total += intersectionArea(p4, p5)

    return total

## does not handle edge case of overlapping circle that are also out of bounds (and overlap is OOB)
def generateSpacingMetric(df, radius): 
    pList = polygonList(df, radius)
    circleArea = generatePointArea(radius)

    totalCourtIntersection = courtIntersection(pList, radius)
    totalPointIntersection = pointsIntersection(pList)

    totalIntersection = totalCourtIntersection + totalPointIntersection

    coveredArea = (circleArea * len(df)) - totalIntersection

    return coveredArea


#%%
full_game_data['team1CASpacing'] = -99
full_game_data['team2CASpacing'] = -99

full_game_data['HOMEDESCRIPTION'] = full_game_data['HOMEDESCRIPTION'].fillna('-')
full_game_data['VISITORDESCRIPTION'] = full_game_data['VISITORDESCRIPTION'].fillna('-')

for idx, row, in tqdm(full_game_data.iterrows(), total=full_game_data.shape[0]): 
    if row['team1CASpacing'] != -99:
        continue 
    else: 
        quarter = row['quarter']
        game_clock = row['game_clock']
        team = row['team_id']
        homedes = row['HOMEDESCRIPTION']
        visdes = row['VISITORDESCRIPTION']

        if team == teamDict['team1']:
            df = full_game_data[(full_game_data.quarter == quarter) & (full_game_data.game_clock == game_clock) & (full_game_data.team_id == teamDict['team1']) & (full_game_data.HOMEDESCRIPTION == homedes) & (full_game_data.VISITORDESCRIPTION == visdes)]

            teamColumn = 'team1CASpacing'
            otherTeamColumn = 'team2CASpacing'

        elif team == teamDict['team2']:
            df = full_game_data[(full_game_data.quarter == quarter) & (full_game_data.game_clock == game_clock) & (full_game_data.team_id == teamDict['team2']) & (full_game_data.HOMEDESCRIPTION == homedes) & (full_game_data.VISITORDESCRIPTION == visdes)]
            teamColumn = 'team2CASpacing'
            otherTeamColumn = 'team1CASpacing'

        else:
            continue

        if len(df) != 5:
            print('DF: ', df)
            print('----------')

            print('quarter :', quarter)
            print('game_clock :', game_clock)

            break

        spacingMetric = generateSpacingMetric(df, 4)

        full_game_data.loc[(full_game_data.quarter == quarter) & (full_game_data.game_clock == game_clock) & (full_game_data.team_id == team), otherTeamColumn] = 0

        full_game_data.loc[(full_game_data.quarter == quarter) & (full_game_data.game_clock == game_clock) & (full_game_data.team_id == team), teamColumn] = spacingMetric


        


# %%
