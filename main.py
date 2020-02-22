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
import sys
import json
from dataset_preprocessor import Preprocessor, GlobalSetting

import warnings; warnings.simplefilter('ignore')



class FoodRecomender:

    def __init__(self, preprocessor=Preprocessor(), settings=GlobalSetting):
        self.settings = settings
        self.preprocessor = preprocessor
        return

    def get_recomendations_by_ingredients(self, meals_ids=None):
        if(self.preprocessor.is_file_exist(
            file_path=self.settings.cosine_matrixes_path,
            file_name='ingredients_cosine_similarity_matrix',
            file_extension='.npy'
        )):
            # Матрица коэффициентов похожести блюд
            cos_sims = self.preprocessor.load_cos_sims(
                filename=os.path.join(self.settings.cosine_matrixes_path, 'ingredients_cosine_similarity_matrix')
            )

        indixes = self.preprocessor.create_indixes('id', self.preprocessor.open_file(GlobalSetting.raw_recepies_path, usecols=['id'], nrows=100))
        if(not meals_ids):
            print('нету мелсов')
        else:
            # Массив идентификаторов похожих блюд
            similar_meals_enumerate = []
            for id in meals_ids:

                # Индекс исходного блюда в косинусной матрице
                cos_sims_el_index = indixes[id]

                # Похожие 5 блюд по убыванию коэффициента
                similarities = list(enumerate(cos_sims[cos_sims_el_index]))
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:6]
                similar_meals_enumerate = sorted(similar_meals_enumerate + similarities, key=lambda x: x[1], reverse=True)


         # TODO: Обработать момент, при котором отсутствует косинусная матрица
         # TODO: Может заранее хранить индексы, а не открывать raw_recipes на чтение, все-таки он большой


        return list(map(lambda x: x[0], similar_meals_enumerate))


    def get_recomendations_by_nutrition(self):
        return


def init(args):
    fr = FoodRecomender()
    comand = args[0]
    payload = json.loads(args[1])
    if comand == '1':
        #meals_ids = list(map(lambda x: x['id'], payload))
        meals_ids = payload
        sys.stdout.write(json.dumps(fr.get_recomendations_by_ingredients(meals_ids=meals_ids)))

init(sys.argv[1:])
