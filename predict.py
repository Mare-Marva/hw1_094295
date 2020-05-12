import argparse
import numpy as np
import pandas as pd

import pickle
import json
import ast


def preprocess_test(data, data_train):
    df_rating = pd.read_csv('rating.tsv', sep='\t')
    columns =[]
    with open('columns.json') as f:
        columns_data = json.load(f)
        for key in columns_data.keys():
            columns = columns_data[key]
            columns.remove('revenue')
            columns.remove('top_actors')
            columns.remove('authors')
            columns.remove('directors')
            columns.remove('screenplay')
            columns.remove('producers')
    df = pd.DataFrame(index =list(range(len(data))),columns=columns)
    data.drop(['poster_path', 'original_title', 'production_countries', 'title', 'status', 'Keywords', 'overview'],
              axis=1, inplace=True)

    # fixing old features
    data['is_image'] = np.where(data['backdrop_path'].isna(), 0, 1)
    data.drop(['backdrop_path'], axis=1, inplace=True)
    data['tagline'] = np.where(data['tagline'].isna(), '', data['tagline'])
    data['is_tagline'] = np.where(data['tagline'].isna(), 0, 1)
    data['homepage'] = np.where(data['homepage'].isna(), 0, 1)
    data['video'] = np.where(data['video'] == False, 0, 1)
    data['runtime'] = data['runtime'].fillna(round(np.mean(data['runtime'].dropna())))

    data['release_date'] = pd.to_datetime(data['release_date'])
    data['month'] = data['release_date'].dt.month
    data['year'] = data['release_date'].dt.year
    data['season_released'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12],
                                     labels=['winter', 'spring', 'summer', 'sutumn']).astype('category')

    num_top_actors =1
    for indx, row in data.iterrows():
        data_list =[]
        data_list.append(row['budget'])
        data_list.append(row['homepage'])
        data_list.append(row['id'])
        data_list.append(row['popularity'])
        data_list.append(row['runtime'])
        data_list.append(row['video'])
        data_list.append(row['vote_average'])
        data_list.append(row['vote_count'])
        data_list.append(row['is_image'])
        data_list.append(row['is_tagline'])

        if row['belongs_to_collection'] == row['belongs_to_collection']: # check that is not NAN
            data_list.append(1)
        else:
            data_list.append(0)

        ## production_companies
        row_companies = ast.literal_eval(row['production_companies'])
        companies = []
        if row_companies != []:
            for company in row_companies:
                companies.append(company['name'])
            company_count = len(row_companies)

            data_list.append(company_count)#companies_num
        else:
            data_list.append(0)  # companies_num

        ## languages
        row_lang = ast.literal_eval(row['spoken_languages'])
        lang_count = len(row_lang)
        data_list.append(lang_count) # languages_num

        try: ## rating from imdb
            data_list.append(float(df_rating[df_rating['tconst'] == data['imdb_id'].iloc[indx]]['averageRating']))
        except:
            data_list.append(0)

        ## genres
        row_genres = ast.literal_eval(row['genres'])
        data_list.append(len(row_genres)) ## genre_num

        ## cast
        top_actor =''
        actors_count = 0
        female_count = 0
        row_cast = ast.literal_eval(row['cast'])
        if row_cast != []:
            actors_count = 0
            female_count = 0
            for actor in row_cast:
                actors_count += 1
                if actor['gender'] == 1:
                    female_count += 1
                if actor['order'] < num_top_actors:
                    top_actor = actor['name']

        data_list.append(len(row_cast)) ## actors_num
        if top_actor in list(data_train['top_actors']):
            data_list.append(data_train[data_train['top_actors'] == top_actor]['actor_rating'].iloc[0])## actor rating
        else:
            data_list.append(0)  ## actor rating
        if actors_count ==0: data_list.append(0) ## female per
        else: data_list.append(female_count / actors_count)



        ## crew
        row_crew = ast.literal_eval(row['crew'])
        author_name, director_name,screenplay_name,producer_name = '','','',''
        crew_count = 0
        if row_crew != []:
            author, director, screenplay, producer = 0, 0, 0, 0
            data_list.append(len(row_crew)) # crew_size
            for crew_member in row_crew:
                crew_count += 1
                if crew_member['department'] == 'Writing' and crew_member['job'] == 'Author' and author == 0:
                    author_name = crew_member['name']
                    author = 1

                if crew_member['department'] == 'Writing' and crew_member['job'] == 'Screenplay' and screenplay == 0:
                    screenplay_name =  crew_member['name']
                    screenplay = 1

                if crew_member['department'] == 'Directing' and crew_member['job'] == 'Director' and director == 0:
                    director_name =  crew_member['name']
                    director = 1

                if crew_member['department'] == 'Production' and crew_member['job'] == 'Producer' and producer == 0:
                    producer_name = crew_member['name']
                    producer = 1

        else:
            data_list.append(0) # crew_size

        ## rating - from test data scraping
        if author_name in list(data_train['authors']):
            data_list.append(data_train[data_train['authors'] == author_name]['author_rating'].iloc[0])## author rating
        else:
            data_list.append(0)

        if director_name in list(data_train['directors']):
            data_list.append(data_train[data_train['directors'] == director_name]['director_rating'].iloc[0])## director rating
        else:
            data_list.append(0)

        if screenplay_name in list(data_train['screenplay']):
            data_list.append(data_train[data_train['screenplay'] == screenplay_name]['screenplay_rating'].iloc[0])## screenplay rating
        else:
            data_list.append(0)

        if producer_name in list(data_train['producers']):
            data_list.append(data_train[data_train['producers'] == producer_name]['producer_rating'].iloc[0])## producer rating
        else:
            data_list.append(0)

        if crew_count != 0 and actors_count!=0:
            data_list.append(crew_count/actors_count)# crew per actor
        else:
            data_list.append(0)

        data_list.append(row['month'])
        data_list.append(row['year'])

        ## one hot encoders

        ## genres encode
        genres = columns[26:44]
        for genre_name in genres:
            if row_genres != []:
                if genre_name.split('_')[1] in row_genres: data_list.append(1)
                else: data_list.append(0)
            else: data_list.append(0)

        # data_list.append(0) ## average budget

        if row['season_released'] == 'spring' : data_list.append(1)
        else: data_list.append(0)
        if row['season_released'] == 'summer' : data_list.append(1)
        else: data_list.append(0)
        if row['season_released'] == 'sutumn' : data_list.append(1)
        else: data_list.append(0)
        if row['season_released'] == 'winter' : data_list.append(1)
        else: data_list.append(0)

        ## company avg rev
        with open('company.json') as f:
            companies = json.load(f)
            for company in companies.keys():
                avg = np.sum(companies[company]) / len(companies[company])
                companies[company].append(avg)


            sum, avg = 0, 0
            if row['production_companies'] != []:
                if len(ast.literal_eval(row['production_companies'])) > 0:
                    for comp in ast.literal_eval(row['production_companies']):
                        if comp['name'] in companies.keys():
                            sum += companies[comp['name']][-1]
                    avg = sum / len(ast.literal_eval(row['production_companies']))
            data_list.append(avg + 1)

        ## names encode
        actors_name = columns[49:79]
        for name in actors_name:
            if name.split('_')[1] == top_actor: data_list.append(1)
            else: data_list.append(0)

        authors_name = columns[79:109]
        for name in authors_name:
            if name.split('_')[1] == author_name: data_list.append(1)
            else: data_list.append(0)

        directors_name =columns[109:139]
        for name in directors_name:
            if name.split('_')[1] == director_name: data_list.append(1)
            else: data_list.append(0)

        screenplay_names = columns[139:169]
        for name in screenplay_names:
            if name.split('_')[1] == screenplay_name: data_list.append(1)
            else: data_list.append(0)

        producers_name = columns[169:]
        for name in producers_name:
            if name.split('_')[1] == producer_name: data_list.append(1)
            else: data_list.append(0)

        df.iloc[indx] = data_list

    # df['AvgSalary'] = df['budget'] / (df['actors_num'] + df['crew_size'])

    df.to_csv('fix_test.csv', header=True, index=False)
    return df


## Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))



if __name__=='__main__':
    df = pd.read_csv('test.tsv', sep='\t')
    df_train = pd.read_csv('fix_train_final5.csv')
    df['revenue'] = np.log(df['revenue'].astype('float64'))
    y = df['revenue']
    preprocess_test(df,df_train)
    test_data = pd.read_csv('fix_test.csv')
    test_data['budget'] = np.log(test_data['budget'].astype('float64'))
    test_data['vote_count'] = np.log(test_data['vote_count'].astype('float64'))

    X = test_data
    X.columns = range(0, 200)

    #####
    model = pickle.load(open("model_xgb.pickle.dat", "rb"))
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)
    print("RMSLE model 1 is: {:.6f}".format(rmsle(np.exp(y), y_pred)))


    model = pickle.load(open("model_lgb.pickle.dat", "rb"))
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)
    print("RMSLE model 2 is: {:.6f}".format(rmsle(np.exp(y), y_pred)))
    predictions = pd.DataFrame({'id': list(X[2]), 'prediction': list(y_pred)})
    predictions.to_csv('prediction.csv', index=False, header=False)
