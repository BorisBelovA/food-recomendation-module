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
                filename=os.path.join(self.settings.cosine_matrixes_path,'ingredients_cosine_similarity_matrix')
            )

        indixes = self.preprocessor.create_indixes('id', self.preprocessor.open_file(GlobalSetting.raw_recepies_path, usecols=['id'], nrows=100))
        if(not meals_ids):
            print('нету мелсов')
        else:
            for id in meals_ids:
                cos_sims_el_index = indixes[id]
                print(cos_sims_el_index)
        return


    def get_recomendations_by_nutrition(self):
        return


def init(args):
    fr = FoodRecomender()
    comand = args[0]
    payload = json.loads(args[1])
    meals_ids = list(map(lambda x: x['id'], payload))
    fr.get_recomendations_by_ingredients(meals_ids=meals_ids)
    #print(meals_ids,  type(payload))

init(sys.argv[1:])

