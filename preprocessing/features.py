# Load games
from __future__ import division

import nflgame
# Extract features
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer



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
                progress = 0
                desc = ''

                # TODO: include sacks? probably not since we can't assign them to any play option
                # TODO: Additonally maybe even booth review, official timeout?
                # TODO: Fumble plays should count as if Fumble didn't happen?
                # TODO: plays with declined penalties should be counted ((4:52) A.Foster right tackle to HOU 43 for 13 yards (J.Cyprien). Penalty on JAC-S.Marks, Defensive Offside, declined.)
                # TODO: plays with accepted penalties that do not nullify the play should be counted (keyword: No Play)
                # TODO: error with group when using 2013
                # TODO: Should we count Def. Pass Interference? Def. Holding?

                if (play.note == None or play.note == 'TD' or play.note == 'INT') \
                        and (' punt' not in play.desc) \
                        and ('END ' != play.desc[:4]) \
                        and ('End ' != play.desc[:4]) \
                        and ('Two-Minute Warning' not in play.desc) \
                        and ('spiked the ball to stop the clock' not in play.desc) \
                        and ('kneels to ' not in play.desc) \
                        and ('Delay of Game' not in play.desc) \
                        and (play.time is not None) \
                        and ('Penalty on' not in play.desc) \
                        and ('Delay of Game' not in play.desc) \
                        and ('sacked at' not in play.desc) \
                        and ('Punt formation' not in play.desc) \
                        and ('Direct snap to' not in play.desc) \
                        and ('Aborted' not in play.desc) \
                        and ('temporary suspension of play' not in play.desc) \
                        and ('TWO-POINT CONVERSION ATTEMPT' not in play.desc) \
                        and ('warned for substitution infraction' not in play.desc) \
                        and ('no play run - clock started' not in play.desc) \
                        and ('challenged the first down ruling' not in play.desc) \
                        and ('*** play under review ***' not in play.desc) \
                        and ('Direct Snap' not in play.desc) \
                        and ('Direct snap' not in play.desc):

                    features['team'] = play.team
                    if play.drive.game.away == play.team:
                        features['opponent'] = play.drive.game.home
                    else:
                        features['opponent'] = play.drive.game.away
                    timeclock = play.time.clock.split(':')

                    features['time'] = float(timeclock[0]) * 60 + float(timeclock[1])
                    if (play.time.qtr == 1) or (play.time.qtr == 3):
                        features['time'] += 15 * 60

                    if play.time.qtr <= 2:
                        features['half'] = 1
                    else:
                        features['half'] = 2

                    features['position'] = 50 - play.yardline.offset
                    features['down'] = play.down
                    features['togo'] = play.yards_togo

                    if 'Shotgun' in play.desc:
                        features['shotgun'] = 1
                    else:
                        features['shotgun'] = 0

                    full_desc = play.desc
                    full_desc = full_desc.replace('No. ', 'No.')
                    while re.search(r" [A-Z]\. ", full_desc) is not None:
                        match = re.search(r" [A-Z]\. ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip())
                    if re.search(r"[^\.] \(Shotgun\)", full_desc) is not None:
                        full_desc = full_desc.replace(" (Shotgun)", ". (Shotgun)")
                    full_desc = full_desc.replace('.(Shotgun)', '. (Shotgun)')

                    if re.search(r" a[st] QB for the \w+ ", full_desc) is not None:
                        match = re.search(r" a[st] QB for the \w+ ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')

                    if re.search(r"New QB.{0,20}[0-9]+ \w+?\.w+? ", full_desc) is not None:
                        match = re.search(r"New QB.{0,20}[0-9]+ \w+?\.w+? ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')

                    if re.search(r"New QB.{0,20}[0-9]+ \w+?[\.\, ] ?\w+? ", full_desc) is not None:
                        match = re.search(r"New QB.{0,20}[0-9]+ \w+?[\.\, ] ?\w+? ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')

                    if re.search(r"\#[0-9]+ Eligible ", full_desc) is not None:
                        match = re.search(r"\#[0-9]+ Eligible ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')

                    full_desc = full_desc.replace('New QB for Denver - No.6 - Brock Osweiler ',
                                                  'New QB for Denver - No.6 - B.Osweiler. ')

                    full_desc = full_desc.replace(' at QB ', ' at QB. ')
                    full_desc = full_desc.replace(' at qb ', ' at QB. ')
                    full_desc = full_desc.replace(' at Qb ', ' at QB. ')
                    full_desc = full_desc.replace(' in as QB for this play ', ' in as QB for this play. ')
                    full_desc = full_desc.replace(' in as QB ', ' in as QB. ')
                    full_desc = full_desc.replace(' in as quarterback ', ' in as QB. ')
                    full_desc = full_desc.replace(' in at Quarterback ', ' in as QB. ')
                    full_desc = full_desc.replace(' is now playing ', ' is now playing. ')
                    full_desc = full_desc.replace(' Seminole Formation ', ' ')
                    full_desc = full_desc.replace(' St. ', ' St.')
                    full_desc = full_desc.replace(' A.Randle El ', ' A.Randle ')
                    full_desc = full_desc.replace('Alex Smith ', 'A.Smith ')

                    if (re.search(r"New QB \#[0-9]+ \w+?\.\w+? ", full_desc) is not None):
                        match = re.search(r"New QB \#[0-9]+ \w+?\.\w+? ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')
                    if (re.search(r"took the reverse handoff from #[0-9]+", full_desc) is not None):
                        match = re.search(r"took the reverse handoff from #[0-9]+ \S+ ", full_desc).group(0)
                        full_desc = full_desc.replace(match, match.rstrip() + '. ')

                    sentences = full_desc.split('. ')
                    flag = 0
                    for i in range(len(sentences)):

                        if ('as eligible (Shotgun) ' in sentences[i]):
                            sentences[i] = re.sub(r"^.+ \(Shotgun\) ", "", sentences[i]).strip()

                        if (re.search(r' eligible \S+\.\S+ ', sentences[i]) is not None):
                            sentences[i] = re.sub(r"^.+ eligible ", "", sentences[i]).strip()

                        if ' as eligible' in sentences[i]:
                            continue

                        if 'was injured during the play' in sentences[i]:
                            continue
                        if 'lines up at ' in sentences[i]:
                            continue

                        if (re.search(r' at QB$', sentences[i]) is not None):
                            continue

                        if ' in at QB' in sentences[i]:
                            sentences[i] = re.sub(r"^.+ in at QB", "", sentences[i]).strip()

                        if ' report as eligible' in sentences[i]:
                            sentences[i] = re.sub(r"^.+ report as eligible", "", sentences[i]).strip()

                        if ('at QB' in sentences[i]) and ('at WR' in sentences[i]):
                            # QB and WR switched positions
                            continue

                        desc = sentences[i]
                        desc = re.sub(r"\(.+?\)", "", desc).strip()
                        desc = re.sub(r"\{.+?\}", "", desc).strip()

                        if ((re.search(r'to \w+$', desc) is not None) or (re.search(r'^\w+$', desc) is not None)) and (
                            i < len(sentences) - 1) and ('respotted to' not in desc):
                            desc = desc + '.' + re.sub(r"\(.+?\)", "", sentences[i + 1]).strip()

                        if ((i < len(sentences) - 1) and (sentences[i + 1][:3] == 'to ')):
                            desc = desc + '.' + re.sub(r"\(.+?\)", "", sentences[i + 1]).strip()

                        if ' at QB' in desc:
                            desc = ''
                            continue
                        if ' eligible' in desc:
                            desc = ''
                            continue

                        if 'Injury update: ' in desc:
                            desc = ''
                            continue
                        if desc.startswith('Reverse') == True:
                            desc = ''
                            continue
                        if desc.startswith('Direction change') == True:
                            desc = ''
                            continue
                        if desc.startswith('Direction Change') == True:
                            desc = ''
                            continue

                        # if (re.search(r'^\S+\.\S+ ', desc) is not None):
                        # if((' pass ' ) in desc) and ((
                        if ' pass ' in desc:
                            if (' short ' in desc) or (' deep' in desc):
                                if (' left' in desc) or (' right' in desc) or (' middle' in desc):
                                    if (' incomplete ' in desc) or (' for ' in desc) or (' INTERCEPTED ' in desc):
                                        break

                        else:
                            if (' up the middle' in desc) or (' left' in desc) or (' right' in desc):
                                if (' for ' in desc):
                                    break
                        desc = ''

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
                            if (play.note != 'INT') and ('INTERCEPTED' not in desc):
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
                                if rematch is None:
                                    print desc
                                    print play.desc
                                offset = 0
                                match = rematch.group(0).split()
                                if match[1] == 'scrambles':
                                    features['qbrun'] = 1
                                    offset = 1

                                if match[2 + offset] == "guard":
                                    features['side'] = 'middle'
                                else:
                                    features['side'] = match[1 + offset]

                        if (play.note == 'INT') or ('INTERCEPTED' in desc):
                            success = 0
                        else:
                            if (play.touchdown == True) and (' fumble' not in play.desc):
                                success = 1
                                success_cnt += 1
                            elif yards >= play.yards_togo:
                                success = 1
                                success_cnt += 1

                            # progress label calculation
                            if yards >= play.yards_togo:
                                # new first down reached
                                progress == 1
                            elif (play.down in [1, 2]) and (yards > 0):
                                progress = (float(yards) / float(play.yards_togo)) ** play.down
                            else:
                                # 3rd or 4th down attempt without conversion
                                progress = 0

                    if features['side'] not in ['middle', 'left', 'right']:
                        print play.desc
                        print
                        continue

                    play_features.append(features)
                    success_labels.append(success)
                    yard_labels.append(yards)
                    progress_labels.append(progress)

    print len(play_features)

    data = {}
    data['features'] = np.array(play_features)
    data['success'] = np.array(success_labels)
    data['yards'] = np.array(yard_labels)
    data['progress'] = np.array(progress_labels)
    data['categorical_features'], data['encoder'] = encode_categorical_features(data['features'], sparse=False)

    return data


# Encode categorical features
def encode_categorical_features(features, sparse=True):
    encoder = DictVectorizer(sparse=sparse)
    encoder.fit(features)
    encoded_features = encoder.transform(features)
    return encoded_features, encoder


def get_team_features(team, features, labels, feature_name='team'):
    team_features = []
    team_labels = []
    for feature_index in range(len(features)):
        if features[feature_index][feature_name] == team:
            f = features[feature_index].copy()
            del f[feature_name]
            team_features.append(f)
            team_labels.append(labels[feature_index])
    print len(team_features), 'features / rows'
    return np.array(team_features), np.array(team_labels)
