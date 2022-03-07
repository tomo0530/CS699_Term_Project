import re
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import KFold
from xfeat import CountEncoder, TargetEncoder


class Data_preprocessing(object):
    filename_pitching = 'https://raw.githubusercontent.com/tomo0530/CS699_Term_Project/master/pitching_info.csv'
    filename_game = 'https://raw.githubusercontent.com/tomo0530/CS699_Term_Project/master/game_info.csv'
    filename_preprocessed = 'baseball_info.csv'

    def __init__(self):
        self.df_pitching, self.df_game = self._read_data_as_df()
        self.df_preprocessed = self.finalize_preprocessing()

    def _read_data_as_df(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            df_pitching = pd.read_csv(Data_preprocessing.filename_pitching)
            df_game = pd.read_csv(Data_preprocessing.filename_game)
            return df_pitching, df_game
        except IOError as e:
            print(e)
            return None

    def preprocess_pitching_data(self) -> pd.DataFrame:
        df = self.df_pitching.copy()

        # delete id and batting result column
        df = df.drop(['id', 'dir', 'dist', 'battingType', 'isOuts'], axis=1)

        # fill R or L from former row in N/A
        df['pitcherHand'] = df['pitcherHand'].interpolate('ffill')
        df['batterHand'] = df['batterHand'].interpolate('ffill')

        # convert type of ball speed and fill average in N/A
        df['speed'] = df['speed'].str.replace('km/h', '').replace('-',
                                                                  '0').astype(
            int)
        sp_ave = int(df.loc[df['speed'] != 0, 'speed'].mean())
        df.loc[df['speed'] == 0, 'speed'] = sp_ave

        # fill pitch type in N/A according to ball speed
        for idx, row in df.iterrows():
            if row['pitchType'] == '-':
                if row['speed'] > sp_ave:
                    df.at[idx, 'pitchType'] = 'ストレート'
                else:
                    df.at[idx, 'pitchType'] = 'スライダー'

        # divide inning into number and top or bottom
        df['inning_number'] = df['inning'].apply(
            lambda x: re.sub(r'[^0-9]', "", x))
        df['inning_tb'] = df['inning'].apply(lambda x: 0 if '表' in x else 1)
        df = df.drop('inning', axis=1)

        # label whether pitcher is batter
        pitcher_df = df.groupby('pitcher').size().reset_index()
        pitcher_df['batter'] = pitcher_df['pitcher']
        pitcher_df['batter_isPitcher'] = 1
        pitcher_df = pitcher_df[['batter', 'batter_isPitcher']]
        df = df.merge(pitcher_df, on='batter', how='left')
        df['batter_isPitcher'] = df['batter_isPitcher'].fillna(0)
        df['batter_isPitcher'] = df['batter_isPitcher'].astype('int')

        # reclassify whether hit or not
        hit = [4, 5, 6, 7]
        df['y'] = df['y'].apply(lambda x: 1 if x in hit else 0)

        return df

    def preprocess_game_data(self) -> pd.DataFrame:
        df = self.df_game.copy()

        # delete unnecessary columns
        df = df.drop(['Unnamed: 0', 'bgBottom', 'bgTop'], axis=1)

        # convert team name into number
        team_list = set(df['topTeam'].unique().tolist() + df[
            'bottomTeam'].unique().tolist())
        team_dict = {team: i for i, team in enumerate(team_list)}
        df['topTeam'] = df['topTeam'].replace(team_dict)
        df['bottomTeam'] = df['bottomTeam'].replace(team_dict)

        # extract time features from startDayTime
        df['startDayTime'] = pd.to_datetime(df['startDayTime'])
        df['minute'] = [i.minute for i in df['startDayTime']]
        df['hour'] = [i.hour for i in df['startDayTime']]
        df['day'] = [i.day for i in df['startDayTime']]
        df['month'] = [i.month for i in df['startDayTime']]
        df['year'] = [i.year for i in df['startDayTime']]
        df['day_of_week'] = [i.dayofweek for i in df['startDayTime']]
        df['day_of_year'] = [i.dayofyear for i in df['startDayTime']]
        df = df.drop(['startDayTime', 'startTime'], axis=1)

        return df

    def merge_pitching_and_game_data(self) -> pd.DataFrame:
        df_pitching = self.preprocess_pitching_data()
        df_game = self.preprocess_game_data()
        df = df_pitching.merge(df_game, on=['gameID'], how='left')
        return df

    def label_pitcher_and_batter_team(self) -> pd.DataFrame:
        df = self.merge_pitching_and_game_data()

        df['pitcher_team'] = df['topTeam']
        df['pitcher_team'] = df['pitcher_team'].where(df['inning_tb'] == 1,
                                                      df['bottomTeam'])

        df['batter_team'] = df['topTeam']
        df['batter_team'] = df['batter_team'].where(df['inning_tb'] == 0,
                                                    df['bottomTeam'])

        return df

    def label_pitcher_and_batter_order(self) -> pd.DataFrame:
        df = self.label_pitcher_and_batter_team()

        df_pitcher = df.groupby(['gameID', 'pitcher_team'])[
            'pitcher'].unique().explode().reset_index()
        df_pitcher['pitcher_order'] = df_pitcher.groupby(
            ['gameID', 'pitcher_team']).cumcount() + 1

        df_batter = df.groupby(['gameID', 'batter_team'])[
            'batter'].unique().explode().reset_index()
        df_batter['batter_order'] = df_batter.groupby(
            ['gameID', 'batter_team']).cumcount() + 1

        df = df.merge(df_pitcher, on=['gameID', 'pitcher_team', 'pitcher'],
                      how='left')
        df = df.merge(df_batter, on=['gameID', 'batter_team', 'batter'],
                      how='left')

        return df

    def finalize_preprocessing(self) -> pd.DataFrame:
        df = self.label_pitcher_and_batter_order()
        df = df.drop('gameID', axis=1)

        # list categorical columns
        categorical_columns = [x for x in df.columns if
                               df[x].dtypes == 'object']

        # count encoding
        count_encoder = CountEncoder(input_cols=categorical_columns)
        df = count_encoder.fit_transform(df)

        # target encoding
        fold = KFold(n_splits=5, shuffle=True, random_state=123)
        target_encoder = TargetEncoder(input_cols=categorical_columns,
                                       target_col='y', fold=fold)
        df = target_encoder.fit_transform(df)

        # drop original object columns
        df = df.drop(categorical_columns, axis=1)

        return df

    def save_preprocessed_data_as_csv(self) -> None:
        self.df_preprocessed.to_csv(
            Data_preprocessing.filename_preprocessed, index=False)


if __name__ == '__main__':
    bb = Data_preprocessing()
    bb.save_preprocessed_data_as_csv()
