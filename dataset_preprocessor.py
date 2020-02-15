import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial.distance import cosine, cdist
import os
from nltk.stem.snowball import SnowballStemmer

import gensim
from gensim.models import Word2Vec

class GlobalSetting:
    current_dir = os.path.dirname(__file__)

    cosine_matrixes_path = os.path.join(current_dir, 'cosine_matrixes')

    raw_recepies_file = 'RAW_recipes.csv'
    raw_recepies_path = os.path.join(current_dir, raw_recepies_file)

    out_file = 'out.csv'
    out_file_extencion = 'csv'
    out_file_path = os.path.join(current_dir, out_file)



class Preprocessor:

    # Конструктор класса
    def __init__(self):
        # Столбцы, которые берутся по умолчанию при открытии файла с данными
        self.standart_recipes_columns = ['name', 'id', 'tags', 'nutrition', 'description', 'ingredients']

        '''if not self.is_file_exist(GlobalSetting.out_file, GlobalSetting.out_file_extencion):
            self.dataframe = self.open_file(GlobalSetting.raw_recepies_path, self.standart_recipes_columns)
        else:
            self.dataframe = self.open_file(GlobalSetting.out_file_path, self.standart_recipes_columns)'''

    # Проверка на наличие выходного файла в каталоге
    def is_file_exist(self, file_path, file_name, file_extension):
        array_of_files = [x for x in os.listdir(file_path) if x.startswith(file_name) & x.endswith(file_extension)]
        return len(array_of_files) > 0

    # Открытие CSV файлв
    def open_file(self, filepath, usecols=None, nrows=None, skiprows=None):
        return pd.read_csv(filepath, usecols=usecols, nrows=nrows, skiprows=skiprows)

    # Срхранение файла
    def save_file(self, dataframe, filename):
        dataframe.to_csv(filename, index=False)

    # Сохранение файла с матрицей похожести
    def save_cosine_sims_matrix(self, filename, cos_sims):
        np.save(os.path.join(GlobalSetting.cosine_matrixes_path, filename), cos_sims)
        print(os.path.join(GlobalSetting.cosine_matrixes_path, filename))

    # Открытие файсла с матрицей
    def load_cos_sims(self, filename):
        return np.load(f'{filename}.npy')

    # Препроцессинг датасета и создание матрицы похожести по ингредиентам
    def preprocess_ingredients(self):
        df = self.open_file(GlobalSetting.raw_recepies_path, usecols=['id', 'ingredients'], nrows=100)
        df['ingredients'] = df['ingredients'].apply(literal_eval)
        stemmer = SnowballStemmer('english')

        # Удаление ингредиентов меньше 3-ч символов и проведение лемматизации
        def t(x):
            arr = []
            for i in x:
                if len(i) > 3:
                    arr.append(stemmer.stem(i))
            return ' '.join(arr)

        df['ingredients'] = df['ingredients'].apply(t)
        vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 1))
        X = vectorizer.fit_transform(df['ingredients'])
        cos_sims = cosine_similarity(X.toarray())
        self.save_cosine_sims_matrix('ingredients_cosine_similarity_matrix', cos_sims)
        return cos_sims

    # Создание векторизированной матрицы
    def create_vectorize_matrix(self, dataframe):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(dataframe)

    # Косинусная матрица близости
    def create_cosine_similarity_matrix(self, *args):
        if len(args) == 2:
            return cosine_similarity(args[0], args[1])
        elif len(args) == 1:
            return cosine_similarity(args[0])

    def create_indixes(self, colname=None, data_frame=None):
        return pd.Series(data_frame.index, index=data_frame[colname])



#p = Preprocessor()
#ing_cos_sims = p.preprocess_ingredients()
#print(ing_cos_sims)

#indices = p.create_indixes('id', p.open_file(GlobalSetting.raw_recepies_path, usecols=['id']))

#print(indices)