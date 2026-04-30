import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

def cleaning(ratings, categories) :
    ratings = ratings.apply(pd.to_numeric, errors='coerce') # If valide str : converts to foalt; else : converts to NaN
    ratings = ratings.dropna() # Droping rows with NaN
    ratings = ratings.astype(int) # Converts float to int
    ratings = ratings[ratings.isin(categories).all(axis=1)] # Keep only rows where all values are valid
    return ratings

def to_fleiss_format(ratings, categories):
    '''Converts ratings to fleiss kappa format'''
    table = []
    for _, row in ratings.iterrows():
        counts = [(row == cat).sum() for cat in categories]
        table.append(counts)
    return np.array(table)

def to_krippendorff_format(ratings) :
    ratings = ratings.to_numpy().T
    return ratings

df = pd.read_csv("../data/annotations_combined.csv") #reading the annotation csv file

pen_ratings = cleaning(df[['pen_1', 'pen_2', 'pen_3', 'pen_4', 'pen_5']],[0,1]) #getting and cleaning only penmarks ratings
hair_ratings = cleaning(df[['hair_1','hair_2','hair_3','hair_4','hair_5']],[0,1,2,3]) #getting and cleaning only hair ratings


#Debuging checks :
# # print("Rows kept:", len(pen_ratings))
# # print("Row sums:", np.unique(fleiss_table.sum(axis=1)))
# # print("First rows of Fleiss table:")
# # print(fleiss_table[:5])

# Computing Fleiss kappa :
pen_fleiss = fleiss_kappa(to_fleiss_format(pen_ratings, categories=[0, 1])) 
hair_fleiss = fleiss_kappa(to_fleiss_format(hair_ratings, categories=[0,1,2,3]))

# Computing Krippendorff kappa :
pen_krippendorff = krippendorff.alpha(reliability_data=to_krippendorff_format(pen_ratings), level_of_measurement='ordinal')
hair_krippendorff = krippendorff.alpha(reliability_data=to_krippendorff_format(hair_ratings), level_of_measurement='ordinal')



print("Fleiss kappa (pen_marks): ", pen_fleiss)
print("Fleiss kappa (hair): ",hair_fleiss)

print("Krippendorff kappa (pen_marks): ",pen_krippendorff)
print("Krippendorff kappa (hair): ", hair_krippendorff)