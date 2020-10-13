from testing import main
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import cm as cm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import sys

def split_string(str):
    
    lst = [int(s) for s in str.split() if s.isdigit()]
    if len(lst) == 0:
        return [0]
    else:
        return lst

def main():

	# General Test for normality
	# print(stats.normaltest(data['audience_average']).pvalue)
	# transf = np.square(data['audience_average'])
	# print(stats.normaltest(transf).pvalue)
	# plt.hist(data['audience_average'])
	#plt.show()
	# We can do log, exp, and other stuff to make it normal. If this happens,
	# we need to do the same for the filtered data
	
	# Import the data
	data1 = pd.read_json('data/omdb-data.json.gz', orient='record', lines=True)
	data2 = pd.read_json('data/rotten-tomatoes.json.gz', orient='record', lines=True)
	data3 = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True, convert_dates=['publication_date'])
	data4 = pd.read_json('data/genres.json.gz', orient='record', lines=True)

	# Merge and join data
	data = data3.join(data2.set_index('rotten_tomatoes_id'), on='rotten_tomatoes_id', lsuffix='', rsuffix='_to_drop').drop(['imdb_id_to_drop'], axis=1) #dropping imdb_id column from rotten tomatoes data because it's incomplete
	data = data.join(data1.set_index('imdb_id'), on='imdb_id')
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
	data = data.drop(columns='filming_location')
	data = data.drop(columns='series')
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

	# filtering to the top 5 genres
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

	#Mann-Whitney: creating finding the means, difference in means, and pvalue by genre
	lst_col = 'genre_label'
	df_genre = pd.DataFrame({col:np.repeat(data[col].values, data[lst_col].str.len())\
			for col in data.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data[lst_col].values)})[data.columns.tolist()]

	genre_pvalue = df_genre.groupby('genre_label')['audience_average'].mean()
	genre_pvalue = genre_pvalue.reset_index()
	genre_pvalue.rename(columns={'audience_average':'audience_average_w_genre'}, inplace=True)

	def mean_wo_genre(genre):
		df_with_genre = df_genre[df_genre['genre_label']==genre]
		w_genre = df_with_genre['audience_average']

		df_wo_genre = data[~data['enwiki_title'].isin(df_with_genre['enwiki_title'].tolist())]
		return df_wo_genre['audience_average'].mean()

	def find_genre_pvalue(genre):
		df_with_genre = df_genre[df_genre['genre_label']==genre]
		w_genre = df_with_genre['audience_average']

		df_wo_genre = data[~data['enwiki_title'].isin(df_with_genre['enwiki_title'].tolist())]
		wo_genre = df_wo_genre['audience_average']
		
		return stats.mannwhitneyu(w_genre, wo_genre).pvalue

	genre_pvalue['audience_average_w/o_genre'] = genre_pvalue['genre_label'].apply(mean_wo_genre)
	genre_pvalue['audience_average_difference'] = genre_pvalue['audience_average_w_genre'] - genre_pvalue['audience_average_w/o_genre']
	genre_pvalue['pvalue'] = genre_pvalue['genre_label'].apply(find_genre_pvalue)
	genre_pvalue = genre_pvalue.sort_values(by=['audience_average_difference'],ascending=False)
	genre_pvalue.to_csv('genre_pvalue.csv', index=False)

    #Mann-Whitney: creating finding the means, difference in means, and pvalue by actor
	lst_col = 'cast_member'
	df2 = pd.DataFrame({col:np.repeat(data[col].values, data[lst_col].str.len())\
	    for col in data.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data[lst_col].values)})[data.columns.tolist()]
	df3 = df2['cast_member'].value_counts()
	df3 = df3.reset_index()
	df3 = df3.loc[df3['cast_member'] >= 30]
	df4 = df2[df2["cast_member"].isin(df3['index'].tolist())]
	df5 = df4.groupby('enwiki_title')['cast_member'].apply(list)
	df5 = df5.reset_index()
	data2 = data.join(df5.set_index('enwiki_title'), on='enwiki_title', lsuffix='_drop', rsuffix='')
	data2 = data2.drop(columns='cast_member_drop')
	data2 = data2[data2.cast_member.notnull()]

    #creating 2 groups
	lst_col = 'cast_member'
	df_actors = pd.DataFrame({col:np.repeat(data2[col].values, data2[lst_col].str.len())\
  		for col in data2.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data2[lst_col].values)})[data2.columns.tolist()]

	actors_pvalue = df_actors.groupby('cast_member')['audience_average'].mean()
	actors_pvalue = actors_pvalue.reset_index()
	actors_pvalue.rename(columns={'audience_average':'audience_average_w_actor'}, inplace=True)

	def mean_wo_actor(actor):
		df_with_actor = df_actors[df_actors['cast_member']==actor]
		w_actor = df_with_actor['audience_average']

		df_wo_actor = data2[~data2['enwiki_title'].isin(df_with_actor['enwiki_title'].tolist())]
		return df_wo_actor['audience_average'].mean()

	def find_actor_pvalue(actor):
		df_with_actor = df_actors[df_actors['cast_member']==actor]
		w_actor = df_with_actor['audience_average']

		df_wo_actor = data2[~data2['enwiki_title'].isin(df_with_actor['enwiki_title'].tolist())]
		wo_actor = df_wo_actor['audience_average']

		return stats.mannwhitneyu(w_actor, wo_actor).pvalue

	actors_pvalue['audience_average_w/o_actor'] = actors_pvalue['cast_member'].apply(mean_wo_actor)
	actors_pvalue['audience_average_difference'] = actors_pvalue['audience_average_w_actor'] - actors_pvalue['audience_average_w/o_actor']
	actors_pvalue['pvalue'] = actors_pvalue['cast_member'].apply(find_actor_pvalue)
	actors_pvalue = actors_pvalue.sort_values(by=['audience_average_difference'],ascending=False)
	actors_pvalue.to_csv('actors_pvalue.csv', index=False)



	#Mann-Whitney: creating finding the means, difference in means, and pvalue by director
	#filtering directors
	lst_col = 'director'
	df12 = pd.DataFrame({col:np.repeat(data[col].values, data[lst_col].str.len())\
	    for col in data.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data[lst_col].values)})[data.columns.tolist()]
	
	df13 = df12['director'].value_counts()
	df13 = df13.reset_index()
	df13 = df13.loc[df13['director'] >= 30]
	
	df14 = df12[df12["director"].isin(df13['index'].tolist())]

	df15 = df14.groupby('enwiki_title')['director'].apply(list)
	df15 = df15.reset_index()
	data2 = data.join(df15.set_index('enwiki_title'), on='enwiki_title', lsuffix='_drop', rsuffix='')
	data2 = data2.drop(columns='director_drop')
	data2 = data2[data2.director.notnull()]
	
	#creating 2 groups
	lst_col = 'director'
	df_directors = pd.DataFrame({col:np.repeat(data2[col].values, data2[lst_col].str.len())\
  		for col in data2.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data2[lst_col].values)})[data2.columns.tolist()]
	
	directors_pvalue = df_directors.groupby('director')['audience_average'].mean()
	directors_pvalue = directors_pvalue.reset_index()
	directors_pvalue.rename(columns={'audience_average':'audience_average_w_director'}, inplace=True)

	def mean_wo_director(director):
		df_with_director = df_directors[df_directors['director']==director]
		w_director = df_with_director['audience_average']

		df_wo_director = data2[~data2['enwiki_title'].isin(df_with_director['enwiki_title'].tolist())]
		return df_wo_director['audience_average'].mean()

	def find_director_pvalue(director):
		df_with_director = df_directors[df_directors['director']==director]
		w_director = df_with_director['audience_average']

		df_wo_director = data2[~data2['enwiki_title'].isin(df_with_director['enwiki_title'].tolist())]
		wo_director = df_wo_director['audience_average']
		return stats.mannwhitneyu(w_director, wo_director).pvalue

	directors_pvalue['audience_average_w/o_director'] = directors_pvalue['director'].apply(mean_wo_director)
	directors_pvalue['audience_average_difference'] = directors_pvalue['audience_average_w_director'] - directors_pvalue['audience_average_w/o_director']
	directors_pvalue['pvalue'] = directors_pvalue['director'].apply(find_director_pvalue)
	directors_pvalue = directors_pvalue.sort_values(by=['audience_average_difference'],ascending=False)
	directors_pvalue.to_csv('directors_pvalue.csv', index=False)

	lst_col = 'director'
	df12 = pd.DataFrame({col:np.repeat(data[col].values, data[lst_col].str.len())\
	    for col in data.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(data[lst_col].values)})[data.columns.tolist()]

	df13 = df12['director'].value_counts()
	df13 = df13.reset_index()
	df13 = df13.loc[df13['director'] >= 30]

	df14 = df12[df12["director"].isin(df13['index'].tolist())]
	df15 = df14.groupby('enwiki_title')['director'].apply(list)
	df15 = df15.reset_index()

	data2 = data.join(df15.set_index('enwiki_title'), on='enwiki_title', lsuffix='_drop', rsuffix='')
	data2 = data2.drop(columns='director_drop')
	data2 = data2[data2.director.notnull()]
	
if __name__ == '__main__':
	main()
