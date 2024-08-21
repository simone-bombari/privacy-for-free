import numpy as np
# import pandas as pd
import csv
import os


def relu(a):
    vec_relu = np.vectorize(lambda x: x * (x > 0))
    return vec_relu(a)

def dev_relu(a):
    vec_dev_relu = np.vectorize(lambda x: float(x > 0))
    return vec_dev_relu(a)



def import_in_df(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(3):
                        new_row.append(int(row[j]))
                    for j in range(3, 7):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['d', 'k', 'N', 'train', 'train_1', 'test', 'test_1']))  # , 'attack', 'attack_1', 'attack_2']))
    df['activation'] = activation
    
    return df

