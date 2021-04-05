#%%
##############################
### IMPORT LIBRARIES, DATA ###
##############################

import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt

## ptd - player tracking data
ptd = pd.read_csv('data/csv/0021500001.csv')
ptd_shortened = ptd.drop_duplicates(subset= ['game_clock', 'shot_clock', 'player_id'], keep='last')
## pbp - play by play
pbp = pd.read_csv('data/events/0021500001.csv')
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
pbp_ptd = ptd_shortened.merge(pbp, left_on = ['game_clock', 'quarter'], right_on = ['game_clock', 'PERIOD'], how = 'outer')


## fill na's with previous valid value, add points scored 
pbp_ptd['SCORE'] = pbp_ptd['SCORE'].fillna(method='ffill')
pbp_ptd['SCOREMARGIN'] = pbp_ptd['SCOREMARGIN'].fillna(method='ffill')
pbp_ptd['SCOREMARGIN'] = pbp_ptd['SCOREMARGIN'].map({'TIE': 0})
pbp_ptd['points_scored'] = abs(pbp_ptd.SCOREMARGIN == pbp_ptd.SCOREMARGIN.shift())



#pbp_ptd['points_scored'] = pbp_ptd['SCOREMARGIN'] - 


#%%
###########################
### ADD RELEVANT FIELDS ###
###########################

## did someone attempt a shot 
pbp_ptd['shot_taken'] = ((pbp_ptd.EVENTMSGTYPE == 1) | (pbp_ptd.EVENTMSGTYPE == 2)).astype('int')

## did someone make a shot
pbp_ptd['shot_made'] = ((pbp_ptd.EVENTMSGTYPE == 1).astype('int'))

## did the offensive player get fouled 
pbp_ptd['foul_reg'] = ((pbp_ptd.EVENTMSGTYPE == 6.0) | (pbp_ptd.EVENTMSGACTIONTYPE != 26.0)).astype('int')






















# %%
## 29 feet top of the arc to baseline 
## 41 from outer circle to baseline 
## 35 for spacing calculations 

###################################
## SPACING METHOD 1: CONVEX HULL ##
###################################
team1 = 1610612737
team2 = 1610612765

playDF = ptd[(ptd.shot_clock ==  8.89) & (ptd.event_id == 14) & ((ptd.team_id == team1) | (ptd.team_id
 == -1))].reset_index()
points = playDF[['x_loc','y_loc']].to_numpy()
hull = ConvexHull(points)
print(hull.volume)

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


# %%
########################################
### SPACING METHOD 2: UNGUARDED AREA ###
########################################


