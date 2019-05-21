import argparse, io, os, re
from os.path import isfile, join
import numpy as np
import pandas as pd 
from sklearn import preprocessing 
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    train_path = args.train_path
    test_path = args.test_path

    train_df, test_df, label_colname = preprocess_images(train_path, test_path)
    classes = separate_classes(train_df, label_colname)
    features_range = range(0, train_df.shape[1]-1)
    
    means = calculate_means_per_class(classes, label_colname, features_range)
    std_devs = calculate_stddevs_per_class(classes, label_colname, features_range)
    labels = train_df[label_colname].unique()
    predictions, actuals = predict(test_df, means, std_devs, labels, label_colname)
    plot_accuracy(predictions, actuals, labels)
    

def normalize_data(df):
    x = df.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def preprocess_images(train_path, test_path, normalize=True):
    label_colname = 'alphabet'

    img_ext = os.listdir(train_path)[0].rpartition('.')[2]
    label_regex = r"([a-z])\d+\." + re.escape(img_ext) + "$"
    label_pattern = re.compile(label_regex) 

    train_paths = [join(train_path,img_path) for img_path in os.listdir(train_path) if isfile(join(train_path, img_path))]
    train_imlist = []
    train_labels = []
    for p in train_paths:
        train_imlist.append( np.asarray(Image.open(io.BytesIO(open(p, 'rb').read())).getdata()) ) 
        train_labels.append( re.search(label_pattern,p).group(1) )

    train_im2Darr = np.asarray(train_imlist)    
    if normalize:
        train_df = pd.DataFrame(train_im2Darr, dtype=float)
        train_df = normalize_data(train_df)
    else: 
        train_df = pd.DataFrame(train_im2Darr)

    train_labels = pd.DataFrame(train_labels, columns=[label_colname])
    train_df = train_df.join(train_labels)
    
    # repeat for test dataframe
    test_paths = [join(test_path,img_path) for img_path in os.listdir(test_path) if isfile(join(test_path, img_path))]

    test_imlist = []
    test_labels = []
    for p in test_paths:
        test_imlist.append( np.asarray(Image.open(io.BytesIO(open(p, 'rb').read())).getdata()) ) 
        test_labels.append( re.search(label_pattern,p).group(1) )

    test_im2Darr = np.asarray(test_imlist)    
    if normalize:
       test_df = pd.DataFrame(test_im2Darr, dtype=float)
       test_df = normalize_data(test_df)
    else: 
        test_df = pd.DataFrame(test_im2Darr)
    
    test_labels = pd.DataFrame(test_labels, columns=[label_colname])
    test_df = test_df.join(test_labels)
    
    return (train_df, test_df, label_colname)

def separate_classes(df, label):
    df = df.sort_values(label, axis=0)
    df_grouped = df.groupby(by=label, axis=0)
    classes = [df_grouped.get_group(gp).reset_index() for gp in df_grouped.groups]

    return classes

def calculate_means_per_class(classes, label, features_range):
    means = dict()
    for gp in classes:
        gp_means = []
        letter = list(gp[label])[0]
        for px in features_range: #-1 to exclude the label col
            gp_means.append( gp[px].mean() )
        means[letter] = gp_means 
    return means 

def calculate_stddevs_per_class(classes, label, features_range):
    # means = calculate_means_per_class(classes, label, features_range)
    std_devs = dict()
    for gp in classes: 
        gp_std_devs = []
        letter = list(gp[label])[0]
        for px in features_range:
            # avg = avg[px]
            # variance = sum( [pow(x-avg,2) for x in gp[px]])/float(gp[px].shape[0]-1)
            variance = gp[px].var()
            stddev = np.sqrt(variance, casting='same_kind')
            gp_std_devs.append(stddev)
        std_devs[letter] = gp_std_devs
    return std_devs

def calculate_gaussian_probability(px, mean, std):
    exponent = np.exp(-(np.power(px - mean, 2) / (2 * np.power(std, 2)) ) )
    prob = (1 / (np.sqrt(2 * np.pi) * std )) * exponent
    
    return prob 

def predict(test_df, means, std_devs, classes, label_colname, min_prob=0.1, balanced=True):
    features = test_df.columns[:-1]
    if balanced: 
    #balanced training set; there is an equal number of images for each class
    #this can be verified by getting a value_count on the dataframe[label]
    #we get 7 for each class
        prob_class = 1/len(classes)
    else:
        prob_class = -1 
    
    predictions = {}
    actuals = {}
    # print(list(test_df.values))

    for example_idx, example in test_df.iterrows():
        results = {}

        for k in classes:
            # p = 0 
            p = 1   
            for pxl in features:
                mean = means[k][pxl]
                std = std_devs[k][pxl]
                # https://stats.stackexchange.com/questions/300262/gaussian-density-function
                # -with-features-that-may-have-zero-standard-deviation
                if std == 0: 
                    # print(tuple([k,pxl]))
                    prob = min_prob
                else:
                    prob = calculate_gaussian_probability(example[pxl], mean, std)
                
                # if prob == 0:
                if prob < min_prob: 
                    prob = min_prob
                    
                p += np.log(prob) #===ln()

            results[k] = np.log(prob_class) + p

        
        max_idx = np.argmax( list(results.values()) )
        predictions[example_idx] = list(results.keys())[max_idx]
        actuals[example_idx] = example[label_colname]
    
    return (predictions, actuals)

def plot_accuracy(predictions, actuals, labels, fname="Accuracy.jpg",dpi=300):
    correct = defaultdict(np.int8)

    for test in predictions:
        if predictions[test] == actuals[test]:
            correct[actuals[test]] += 1
    
    slabels = sorted(labels)
    scount = [correct[l] for l in slabels]

    plt.scatter(slabels, scount, s=10, c='m')
    plt.xlabel("Class (Character)")
    plt.ylabel("Count of correctly-predicted Images per class")

    plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description='Arguments Parser')

    parser.add_argument('--train-path', action="store", help="relative path to directory of images used for training",
    nargs="?", metavar="train_path")

    parser.add_argument('--test-path', action="store", help="relative path to directory of images used for testing", 
    nargs="?", metavar="test_path")

    
    args = parser.parse_args()
main()
