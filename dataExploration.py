#%%
##############################
### IMPORT LIBRARIES, DATA ###
##############################

import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import math

def x_round(x):
    return math.floor(x*4)/4

## ptd - player tracking data
ptd = pd.read_csv('data/csv/0021500001.csv')
ptd['game_clock'] = ptd.apply(lambda row: x_round(row['game_clock']), axis=1)
ptd_shortened = ptd.drop_duplicates(subset= ['game_clock', 'quarter', 'player_id'], keep='last')

## pbp - play by play
pbp = pd.read_csv('data/events/0021500001.csv')
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
            subset_df = full_game_data[(full_game_data.team_id == row['team_id']) & (full_game_data.game_clock == row['game_clock']) & (full_game_data.shot_clock == row['shot_clock']) & (full_game_data.quarter == row['quarter'])] 
            
            hullSpacingVal = hullSpacing(subset_df)

            full_game_data.loc[ ( (full_game_data['game_clock'] == row['game_clock']) & (full_game_data['shot_clock'] == row['shot_clock'] )  &  (full_game_data['quarter'] == row['quarter'])  ) , 'team1_hullSpacing'] = hullSpacingVal

            print(full_game_data['team1_hullSpacing'])

    elif team == 'team2':
        if row['team2_hullSpacing'] > 0:
            pass
        else:
            subset_df = full_game_data[(full_game_data.team_id == row['team_id']) & (full_game_data.game_clock == row['game_clock']) & (full_game_data.shot_clock == row['shot_clock']) & (full_game_data.quarter == row['quarter'])] 
            
            hullSpacingVal = hullSpacing(subset_df)

            full_game_data.loc[ ( (full_game_data['game_clock'] == row['game_clock']) & (full_game_data['shot_clock'] == row['shot_clock'] )  &  (full_game_data['quarter'] == row['quarter'])  ) , 'team2_hullSpacing'] = hullSpacingVal


full_game_data.apply(lambda row: computeHullSpacing(row), axis=1)




## Plot points 
# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


# %%
########################################
### SPACING METHOD 2: UNGUARDED AREA ###
########################################


