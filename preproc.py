import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


# Split awards
def split_string(str):
    """
    Splits str to a 
    """
    lst = [int(s) for s in str.split() if s.isdigit()]
    if len(lst) == 0:
        return [0]
    else:
        return lst

def main():
    """
    Main Function to import and preprocess the data
    
    """
    # Import the data
    data1 = pd.read_json('data/omdb-data.json.gz', orient='record', lines=True)
    data2 = pd.read_json('data/rotten-tomatoes.json.gz', orient='record', lines=True)
    data3 = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True, convert_dates=['publication_date'])
    data4 = pd.read_json('data/genres.json.gz', orient='record', lines=True)
    
    # Merge and join data
    data = data3.join(data2.set_index('rotten_tomatoes_id'), on='rotten_tomatoes_id', lsuffix='', rsuffix='_to_drop').drop(['imdb_id_to_drop'], axis=1) #dropping imdb_id column from rotten tomatoes data because it's incomplete
    data = data.join(data1.set_index('imdb_id'), on='imdb_id')
    # Need to extract the first item of genre to use wikidata match
    #data["new_genre"] = data["genre"].str[0]
    #data = data.join(data4.set_index('wikidata_id'), on='new_genre')
    
    # Drop what we don't need
    data = data[data.audience_average.notnull() & data.publication_date.notnull()].drop(columns=['based_on', 'omdb_plot', 'made_profit', 'label', 'rotten_tomatoes_id', 'metacritic_id', 'omdb_genres'])
    
    nan = pd.DataFrame(data.isna().sum()/len(data)*100, columns=['percent_nan'])
    nan = nan.reset_index()
    nan = nan.rename(columns={"index": "columns"})
    nan.to_csv('nan.csv', index=False)
    # For the awards, we want to set nan to 0 (no award, no nominations)
    # and find numbers and put them into a list
    data['omdb_award'] = data['omdb_awards'].fillna('0', inplace=True)
    data['award'] = data['omdb_awards'].apply(split_string)
    
    # For the awards and nominations we want to split into 3 columns: oscars, awards, nominations
    df2 = data['award'].apply(pd.Series)
    df2 = df2.rename(columns={0: "oscars", 1: "awards", 2: "nominations"})
    df2 = df2.fillna(0)
    df2 = df2.astype(int)       # it was in a float instead of an int format
    data = pd.concat([data, df2], axis=1)
    # Don't need this data anymore
    del df2
    
    # Drop more useless stuff that either has too many NaN or is not usefull numerically
    data = data.drop(columns='award')
    data = data.drop(columns='omdb_award')
    data = data.drop(columns='omdb_awards')
    data = data.drop(columns='main_subject')
    # data = data.drop(columns='genre')
    data = data.drop(columns='filming_location')
    data = data.drop(columns='series')
    #data = data.drop(columns='new_genre')
    #data = data.drop(columns='wikidata_id')
    data = data.drop(columns='imdb_id')
    
    # Get the date stuff in a useful format
    data['year'] = data['publication_date'].dt.year
    data['month'] = data['publication_date'].dt.month
    data['day'] = data['publication_date'].dt.day
    data['dayofweek'] = data['publication_date'].dt.dayofweek

    # Just assume NaN original_languages are in english
    data['original_language'] = data['original_language'].fillna('Q1860')
    # Just assume NaN country_of_origin are in USA
    data['country_of_origin'] = data['country_of_origin'].fillna('Q30')

    # Fill The director column with a new keyword for NaN
    data['director'].loc[data['director'].isnull()] = data['director'].loc[data['director'].isnull()].apply(lambda x: ['QNULL'])
    #data["director"] = data["director"].str[0]
    
    # Fill The cast column with a new keyword for NaN. we can assume that there
    # will be at least 1 actor per movie
    data['cast_member'].loc[data['cast_member'].isnull()] = data['cast_member'].loc[data['cast_member'].isnull()].apply(lambda x: ['QNULL'])
    # Count the number of actors in the cast and create a new column. This will
    # Preserve some information but make it usable.
    data['num_actors'] = data['cast_member'].str.len()
    
    # Better to just drop
    # We have a lot of NaN in the critic_average and critic_percent columns.

    lst_col = 'genre'
    df6 = pd.DataFrame({col:np.repeat(data[col].values, data[lst_col].str.len())\
        for col in data.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data[lst_col].values)})[data.columns.tolist()]
    df7 = df6['genre'].value_counts(normalize=True) * 100
    df7 = df7.reset_index()
    df7 = df7.loc[df7['genre'] >= 3.4]
    df8 = df6[df6["genre"].isin(df7['index'].tolist())]
    df8 = df8.join(data4.set_index('wikidata_id'), on='genre')
    df8 = df8.drop(columns = 'genre')
    df9 = df8.groupby('enwiki_title')['genre_label'].apply(list)
    df9 = df9.reset_index()
    data = data.join(df9.set_index('enwiki_title'), on='enwiki_title', lsuffix='_drop', rsuffix='')
    #data = data.drop(columns='cast_member_drop')
    data = data[data.genre_label.notnull()]
    data = data.drop(columns='wikidata_id')
    data.to_csv('stats.csv', index=False)
    
    # 1 Hot encode genres
    mlb = MultiLabelBinarizer()
    data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('genre_label')),
                            columns=mlb.classes_,
                            index=data.index))
    data['cast_member'] = data['cast_member'].apply(', '.join)
    data['director'] = data['director'].apply(', '.join)
    
    # We prefered to 1 hot encode over label encde these categories
    # but there are too many columns, it takes too long to process
    le = LabelEncoder()
    data['cast_member'] = le.fit_transform(data.cast_member.values)
    data['director'] = le.fit_transform(data.director.values)
    data['country_of_origin'] = le.fit_transform(data.country_of_origin.values)
    data['original_language'] = le.fit_transform(data.original_language.values)

    # Drop this stuff
    data = data.drop(columns='genre')
    data = data.drop(columns='enwiki_title')
    # Convert the data to something we can use
    data['publication_date'] = data.publication_date.values.astype(np.int64) // 10 ** 9
    data = data.dropna()

    data.to_csv('ml.csv', index=False)


    
    # We can use this in the report to explain the # of columns with a lot of NAN
    # nan = pd.DataFrame(data.isna().sum()/len(data)*100, columns=['percent_nan'])
    # nan = nan.reset_index()
    # nan = nan.rename(columns={"index": "columns"})
    # print(nan.head(26))

if __name__ == '__main__':
    main()