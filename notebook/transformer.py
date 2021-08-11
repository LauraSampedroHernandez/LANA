import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MerchantStateEncoder(BaseEstimator, TransformerMixin):

   def __init__(self):
       pass

   def fit(self, X, y=None):
       return self

   def transform(self, X):
       X = X.copy()
       X = np.where(X == 'CA', 1,
           np.where(X == 'TX', 2,
           np.where(X == 'OH', 3,
           np.where(X == 'NY', 4,
           np.where(X == 'NJ', 5,
           np.where(X == 'FL', 6,
           np.where(X == 'MI', 7,
           np.where(X == 'IN', 8,
           np.where(X == 'NC', 9,
           np.where(X == 'IL', 10,
           np.where(X == 'VA', 11,
           np.where(X == 'GA', 12,
           np.where(X == 'OK', 13,
           np.where(X == 'SC', 14,
           np.where(X == 'MO', 15,
           np.where(X == 'AL', 16,
           np.where(X == 'MN', 17,
           np.where(X == 'TN', 18,
           np.where(X == 'PA', 19,
           np.where(X == 'WA', 20,
           np.where(X == 'MD', 21,
           np.where(X == 'MA', 22,
           np.where(X == 'WI', 23,
           np.where(X == 'CO', 24,
           np.where(X == 'LA', 25,
           np.where(X == 'AZ', 26,
           np.where(X == 'OR', 27,
           np.where(X == 'MS', 28,
           np.where(X == 'KY', 29,
           np.where(X == 'HI', 30,
           np.where(X == 'IA', 31,
           np.where(X == 'AR', 32,
           np.where(X == 'SD', 33,
           np.where(X == 'UT', 34,
           np.where(X == 'NV', 35,
           np.where(X == 'KS', 36,
           np.where(X == 'NM', 37,
           np.where(X == 'CT', 38,
           np.where(X == 'VT', 39,
           np.where(X == 'NE', 40,
           np.where(X == 'NH', 41,
           np.where(X == 'ID', 42,
           np.where(X == 'RI', 43,
           np.where(X == 'ME', 44,
           np.where(X == 'WY', 45,
           np.where(X == 'MT', 46,
           np.where(X == 'DE', 47,
           np.where(X == 'ND', 48,
           np.where(X == 'AK', 49,
           np.where(X == 'DC', 50,
           np.where(X == 'WV', 51,
           np.where(X == 'AA', 52,
           53))))))))))))))))))))))))))))))))))))))))))))))))))))

       return X


class UseChipEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass              

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = np.where(X == 'Chip Transaction', 1,
            np.where(X == 'Swipe Transaction', 2,
            3))
        return X


class MCC_encoder(BaseEstimator, TransformerMixin): 

    def __init__(self):
        pass      

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        list_cluster_1  =  [7922, 7011, 5651, 8043, 
                            5655, 7995, 5621, 5661, 
                            5921, 7542, 5310, 5300, 
                            5813, 8099, 7210, 5211, 
                            7832, 4121, 8011, 8021,
                            5912, 5947, 5970, 5541, 
                            6300, 7276, 7393, 7802, 
                            8041, 8049, 8931, 5977, 
                            5311, 5533, 3144, 3395, 
                            3393, 3390, 3389, 8111,
                            3387, 3359, 3260, 3256, 
                            3174, 3132, 5499, 3075, 
                            3066, 3058, 3009, 3008, 
                            3007, 3006, 3005, 3001, 
                            3405, 3504, 3509, 3596,
                            3000, 5261, 5251, 5192, 
                            5045, 4900, 4829, 4814, 
                            4214, 4131, 4112, 4111, 
                            3780, 3775, 3771, 3730, 
                            3722, 3684, 3640, 9402]

        list_cluster_2  =  [7996, 5094, 5193, 7531, 
                            5941, 8062, 5712, 5932, 
                            7549]

        list_cluster_3  =  [5733, 5722]

        X = np.where(np.isin(X, list_cluster_1), 1 ,
            np.where(np.isin(X, list_cluster_2), 2 ,
            np.where(np.isin(X, list_cluster_3), 3 ,
            4)))
            
        return X   




