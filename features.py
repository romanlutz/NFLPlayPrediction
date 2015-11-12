#Load games
from __future__ import division
import nflgame

# Extract features
import re
from collections import defaultdict
import numpy as np


def extract_features(start_year, end_year):
    play_features = []
    success_labels = []
    yard_labels = []
    progress_labels = []
    success_cnt = 0

    for year in range(start_year, end_year + 1):
        # split into individual weeks in order to avoid having to load
        # large chunks of data at once
        for week in range(1, 18):
            games = nflgame.games(year, week=week)

            for play in nflgame.combine_plays(games):
                features = defaultdict(float)
                success = 0
                yards = 0
                desc = ''

                # TODO: include sacks? probably not since we can't assign them to any play option
                # TODO: Additonally maybe even booth review, official timeout?
                # TODO: Fumble plays should count as if Fumble didn't happen?
                # TODO: plays with declined penalties should be counted ((4:52) A.Foster right tackle to HOU 43 for 13 yards (J.Cyprien). Penalty on JAC-S.Marks, Defensive Offside, declined.)
                # TODO: plays with accepted penalties that do not nullify the play should be counted (keyword: No Play)
                # TODO: error with group when using 2013
                # TODO: Should we count Def. Pass Interference? Def. Holding?

                if (play.note == None or play.note == 'TD' or play.note =='INT') \
                    and (' punt' not in play.desc) \
                    and ('END ' != play.desc[:4]) \
                    and ('End ' != play.desc[:4]) \
                    and ('Two-Minute Warning' not in play.desc) \
                    and ('spiked the ball to stop the clock' not in play.desc) \
                    and ('kneels to ' not in play.desc) \
                    and ('Delay of Game' not in play.desc)\
                    and (play.time is not None)\
                    and ('Penalty on' not in play.desc)\
                    and ('Delay of Game' not in play.desc)\
                    and ('sacked at' not in play.desc)\
                    and ('Punt formation' not in play.desc)\
                    and ('Direct snap to' not in play.desc)\
                    and ('Aborted' not in play.desc)\
                    and ('temporary suspension of play' not in play.desc):

                    features['team'] = play.team
                    if play.drive.game.away == play.team:
                        features['opponent'] = play.drive.game.home
                    else:
                        features['opponent'] = play.drive.game.away
                    timeclock = play.time.clock.split(':')
                    
                    features['time'] = float(timeclock[0])*60 + float(timeclock[1])
                    if (play.time.qtr == 1) or (play.time.qtr == 3):
                        features['time'] += 15*60                        
                 
                    if play.time.qtr <= 2:
                        features['half'] = 1
                    else:
                        features['half'] = 2
                    
                    features['position'] = 50-play.yardline.offset
                    features['down'] = play.down
                    features['togo'] = play.yards_togo

                    if 'Shotgun' in play.desc:
                        features['shotgun'] = 1

                    full_desc = play.desc
                    full_desc = full_desc.replace('No. ','No.')
                    while (re.search(r" [A-Z]\. ", full_desc) is not None):
                        match = re.search(r" [A-Z]\. ", full_desc).group(0)
                        full_desc = full_desc.replace(match,match.rstrip())
                        
                    sentences = full_desc.split('. ')
                    for i in range(len(sentences)):                            
                        if ('as eligible (Shotgun) ' in sentences[i]):
                            sentences[i] = re.sub(r"^.+ \(Shotgun\) ", "", sentences[i]).strip()
  
                        if (re.search(r' eligible \S+\.\S+ ', sentences[i]) is not None):
                            sentences[i] = re.sub(r"^.+ eligible ", "", sentences[i]).strip()
                            
                        if ' as eligible' in sentences[i]:
                            continue       
                            
                        if 'was injured during the play' in sentences[i]:
                            continue
                            

                        if (re.search(r' at QB$', sentences[i]) is not None):
                            continue

                        if ' in at QB' in sentences[i]:
                            sentences[i] = re.sub(r"^.+ in at QB", "", sentences[i]).strip()
                            
                        if ' report as eligible' in sentences[i]:
                            sentences[i] = re.sub(r"^.+ report as eligible", "", sentences[i]).strip()
                            
                        if ('at QB' in sentences[i]) and ('at WR' in sentences[i]):
                            #QB and WR switched positions
                            continue

                        desc = sentences[i]
                        desc = re.sub(r"\(.+?\)", "", desc).strip()
                        
                        if ((re.search(r'to \w+$', desc) is not None) or (re.search(r'^\w+$', desc) is not None)) and (i<len(sentences)-1):
                            desc = desc + '.' + re.sub(r"\(.+?\)", "", sentences[i+1]).strip()
                            

                        if ((i<len(sentences)-1) and (sentences[i+1][:3] == 'to ')):
                            desc = desc + '.' + re.sub(r"\(.+?\)", "", sentences[i+1]).strip()
                            
                        
                            
                        if ' in at QB' in desc:
                            desc = ''
                            continue
                        if ' eligible' in desc:
                            desc = ''
                            continue

                        if (re.search(r'^\S+\.\S+ ', desc) is not None):
                            break
                    
                    if desc == '':
                        continue
                        

                    if 'incomplete' in desc:
                        features['pass'] = 1
                        rematch = re.search(r'incomplete \S+ \S+ to ', desc)

                        if rematch is None:
                            # ball just thrown away, no intended target -> ignore
                            continue;

                        match = rematch.group(0).split()
                        features['passlen'] = match[1]
                        features['side'] = match[2]
                    else:
                        if 'no gain' in desc:
                            yards = 0
                        else:
                            if (play.note!='INT') and ('INTERCEPTED' not in desc):
                                rematch = re.search(r'[-]?[0-9]+ yard\s?', desc)
                                if rematch is None:
                                    print desc
                                    print play.desc
                                match = rematch.group(0)
                                yards = float(match[:match.find(' ')])

                        if ' pass ' in desc:
                            features['pass'] = 1
                            match = re.search(r'pass \S+ \S+', desc).group(0).split()
                            if match[1] == 'to':
                                continue
                            features['passlen'] = match[1]
                            features['side'] = match[2]
                        else:
                            features['pass'] = 0
                            if 'up the middle' in desc:
                                features['side'] = 'middle'
                            else:               
                                rematch = re.search(r'^\S+ (scrambles )?\S+ \S+', desc) 
                                offset = 0
                                match = rematch.group(0).split()
                                if match[1] == 'scrambles':
                                    features['qbrun'] = 1
                                    offset = 1

                                if match[2+offset] == "guard":
                                    features['side'] = 'middle'
                                else:
                                    features['side'] = match[1+offset]

                        if (play.note=='INT') or ('INTERCEPTED' in desc) :
                            success = 0
                        else:
                            if (play.touchdown == True) and (' fumble' not in play.desc):
                                success = 1
                                success_cnt += 1
                            elif yards >= play.yards_togo:
                                success = 1
                                success_cnt += 1

                            # progress label calculation
                            if yards < play.yards_togo:
                                if play.down > 2:
                                    progress = 0
                                elif play.down == 2:
                                    progress = float(yards) / float(play.yards_togo)
                                else: # 1st down - two attempts left
                                    progress = float(yards) * 2.0 / float(play.yards_togo)
                            else:
                                progress = 1 + float(yards - play.yards_togo) / 10.0

                    play_features.append(features)
                    success_labels.append(success)
                    yard_labels.append(yards)
                    progress_labels.append(progress)

                # Debug information
                #import random
                #if random.randint(0,1000) < 2:
                #    print desc
                #    print play.desc
                #    print features
                #    print 'SUCCESS:',success,'| YARDS:',yards
                #    print "############################################################"

                '''
                # Some debug code (Roman)
                else:
                    if 'Timeout' not in play.desc and \
                                    'kicks' not in play.desc and \
                                    'kneels' not in play.desc and \
                                    'Field Goal' not in play.desc and\
                                    'field goal' not in play.desc and\
                                    'Two-Minute-Warning' not in play.desc and \
                                    'END' not in play.desc and\
                                    'Two-Point' not in play.desc and\
                                    'TWO-POINT' not in play.desc and\
                                    'Two-Minute' not in play.desc and\
                                    'punts' not in play.desc and\
                                    'Punt' not in play.desc and\
                                    'spiked' not in play.desc and\
                                    'extra point' not in play.desc and\
                                    'False Start' not in play.desc and\
                                    'Delay of Game' not in play.desc and\
                                    'No Play' not in play.desc and\
                                    'BLOCKED' not in play.desc and\
                                    'FUMBLES' not in play.desc and\
                                    'sacked' not in play.desc:
                        print play.desc
                '''

    print len(play_features)

    return np.array(play_features), np.array(success_labels), np.array(yard_labels), np.array(progress_labels)


# Encode categorical features
# Returns encoded features and the encoder
def encode_categorical_features(features, sparse=True):
    from sklearn.feature_extraction import DictVectorizer

    enc = DictVectorizer(sparse=sparse)
    enc.fit(features)  
    svm_features = enc.transform(features)
    return svm_features, enc
