import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize']=(10, 10)


col_name = [
    "timedelta", # Days between the article publication and the dataset acquisition (non-predictive)
    "n_tokens_title", #  Number of words in the title
    "n_tokens_content", #  Number of words in the content
    "n_unique_tokens", # Rate of unique words in the content
    "n_non_stop_words", #  Rate of non-stop words in the content
    "n_non_stop_unique_tokens", # Rate of unique non-stop words in the content
    "num_hrefs", # Number of links
    "num_self_hrefs", # Number of links to other articles published by Mashable
    "num_imgs", # Number of images
    "num_videos", # Number of videos
    "average_token_length", #  Average length of the words in the content
    "num_keywords", # Number of keywords in the metadata
    "data_channel_is_lifestyle", #  Is data channel 'Lifestyle'?
    "data_channel_is_entertainment", #  Is data channel 'Entertainment'?
    "data_channel_is_bus", #  Is data channel 'Business'?
    "data_channel_is_socmed", #  Is data channel 'Social Media'?
    "data_channel_is_tech", # Is data channel 'Tech'?
    "data_channel_is_world", # Is data channel 'World'?
    "kw_min_min", #  Worst keyword (min. shares)
    "kw_max_min", # Worst keyword (max. shares)
    "kw_avg_min", # Worst keyword (avg. shares)
    "kw_min_max", # Best keyword (min. shares)
    "kw_max_max", # Best keyword (max. shares)
    "kw_avg_max", # Best keyword (avg. shares)
    "kw_min_avg", # Avg. keyword (min. shares)
    "kw_max_avg", # Avg. keyword (max. shares)
    "kw_avg_avg", # Avg. keyword (avg. shares)
    "self_reference_min_shares", # Min. shares of referenced articles in Mashable
    "self_reference_max_shares", # Max. shares of referenced articles in Mashable
    "self_reference_avg_sharess",  # Avg. shares of referenced articles in Mashable
    "weekday_is_monday", # Was the article published on a Monday?
    "weekday_is_tuesday", # Was the article published on a Tuesday?
    "weekday_is_wednesday", # Was the article published on a Wednesday?
    "weekday_is_thursday", # Was the article published on a Thursday?
    "weekday_is_friday", # Was the article published on a Friday?
    "weekday_is_saturday", # Was the article published on a Saturday?
    "weekday_is_sunday", # Was the article published on a Sunday?
    "is_weekend", # Was the article published on the weekend?
    "LDA_00", # Closeness to LDA topic 0
    "LDA_01", # Closeness to LDA topic 1
    "LDA_02", # Closeness to LDA topic 2
    "LDA_03", # Closeness to LDA topic 3
    "LDA_04", # Closeness to LDA topic 4
    "global_subjectivity", # Text subjectivity
    "global_sentiment_polarity", #  Text sentiment polarity
    "global_rate_positive_words", # Rate of positive words in the content
    "global_rate_negative_words", # Rate of negative words in the content
    "rate_positive_words", # Rate of positive words among non-neutral tokens
    "rate_negative_words", # Rate of negative words among non-neutral tokens
    "avg_positive_polarity", #  Avg. polarity of positive words
    "min_positive_polarity", #  Min. polarity of positive words
    "max_positive_polarity", # Max. polarity of positive words
    "avg_negative_polarity", # Avg. polarity of negative words
    "min_negative_polarity", # Min. polarity of negative words
    "max_negative_polarity", # Max. polarity of negative words
    "title_subjectivity", # Title subjectivity
    "title_sentiment_polarity", # Title polarity
    "abs_title_subjectivity", # Absolute subjectivity level
    "abs_title_sentiment_polarity", # Absolute polarity level
]


def find_index(alist, elem) -> int:
    for i, name in enumerate(alist):
        if name == elem:
            return i
    return -1


def is_int(number):
    if abs(float(int(number)) - number) < 1e-6:
        return True
    else:
        return False


def preprocess(data):
    idxes = []
    for i, feature_name in enumerate(data["col_names"]):
        print("")
        print(feature_name)
        data_typed = data["data"][:, i]
        print("data max: ", data_typed.max())
        print("data min: ", data_typed.min())
        # print(f"argmax line: {data['data'][data_typed.argmax(), :]}")
        print("90 percentile: ", np.percentile(np.sort(data_typed), 90))
        print("10 percentile: ", np.percentile(np.sort(data_typed), 10))
        print("99 percentile: ", np.percentile(np.sort(data_typed), 99))
        print("mean: ", np.mean(data_typed))
        print("std: ", np.std(data_typed))
        lower_bound = np.mean(data_typed) - 3. * np.std(data_typed)
        upper_bound = np.mean(data_typed) + 3. * np.std(data_typed)
        print("lower bound: ", lower_bound)
        print("upper bound: ", upper_bound)
        # if np.percentile(np.sort(data_typed), 90) != np.percentile(np.sort(data_typed), 10):
        #     print(np.where(np.abs(data_typed) > 100 * np.percentile(np.sort(np.abs(data_typed)), 90))[0].tolist()[:100])
        #     idxes.extend(np.where(np.abs(data_typed) > 100 * np.percentile(np.sort(np.abs(data_typed)), 90))[0].tolist())
            # idxes.extend(np.where(np.abs(data_typed)+1e-2 < 0.01 * np.percentile(np.sort(np.abs(data_typed)), 10))[0].tolist())
        if lower_bound != upper_bound:
            tmp1 = np.where(data_typed > upper_bound)[0].tolist()
            tmp2 = np.where(data_typed < lower_bound)[0].tolist()
            tmp3 = np.where(data_typed > np.percentile(np.sort(np.abs(data_typed)), 99))[0].tolist()
            print(tmp1, tmp2, tmp3)
            print(f"count (N): {len(tmp1) + len(tmp2)}")
            print(f"count (gamma): {len(tmp3)}")
            print(f"count (<0): {len(np.where(data_typed < 0.0)[0].tolist())}")
            idxes.extend(tmp1)
            idxes.extend(tmp2)
    idxes = np.unique(idxes)
    print(idxes)
    print(len(idxes))
# invalid: 5052 12972 12985 22190 22892 24754
    # data["data"] = np.delete(data["data"], 24754, axis=0)
    # data["label"] = np.delete(data["label"], 24754, axis=0)
    # for idx in idxes:
    #     print(f"data {idx}:")
    #     for i, feature_name in enumerate(data["col_names"]):
    #         print(f'\t{i}-{feature_name}: {data["data"][idx, i]}')
    return data


# 分类的平衡性
def feature_distribution(data):
    for i, feature_name in enumerate(data["col_names"]):
        if is_int(data["data"][0, i]) and is_int(data["data"][1, i]):
            type_is_int = True
            data_typed = np.array(data["data"][:, i], dtype=int)
        else:
            type_is_int = False
            data_typed = data["data"][:, i]
        
        # sns.set() # for style
        sns.displot( 
            data=data_typed,
            kind="hist",
            kde=not type_is_int,
            bins=min(45, int(data_typed.max() - data_typed.min()))+1 if type_is_int else 100,
        )
        plt.title(f"Distributions over `{feature_name}`", y=-0.11)
        # plt.show()
        plt.savefig(f"./fig/dist-{feature_name}.png")


# 样本特征的相关性
def feature_corr(data):
    L = len(data["col_names"])
    corr_mat = np.zeros((L, L))
    for i, feature_name_i in enumerate(data["col_names"]):
        for j, feature_name_j in enumerate(data["col_names"]):
            data_i = data["data"][:, i]
            data_j = data["data"][:, j]
            rho = np.corrcoef(data_i, data_j)[0][1] # pearson
            corr_mat[i][j] = rho
            if i < j and abs(rho) > 0.75:
                print(f"corr `{feature_name_i}`<->`{feature_name_j}` = {rho}")
    sns.heatmap(data=corr_mat, square=True, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    plt.title(f"Pearson Corr Heatmap between features", y=-0.11)
    plt.show()


# 特征和标签的相关性
def feature_label_corr(data):
    L = len(data["col_names"])
    corrs = []
    for i, feature_name_i in enumerate(data["col_names"]):
        data_i = data["data"][:, i]
        label = data["label"]
        rho = np.corrcoef(data_i, label)[0][1] # pearson
        corrs.append(rho)
        # print(f"corr `{feature_name_i}`<->label = {rho}")
    plt.plot()
    plt.bar(list(range(L)), corrs)
    plt.title(f"Pearson Corr between feature and label", y=-0.11)
    plt.show()


# `is` 类的分类统计
def theme_stat(data):
    theme_count = []
    for name in [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]:
        idx = find_index(data["col_names"], name)
        _data = np.array(data["data"][:, idx], dtype=int)
        theme_count.append(sum(_data))
    print(theme_count)
    theme_count.append(len(data["data"]) - sum(theme_count))
    X = [
        "Lifestyle",
        "Entertainment",
        "Business",
        "Social Media",
        "Tech",
        "World",
        "others",
    ]
    Y = theme_count
    plt.plot()
    plt.bar(x=X, height=Y)
    for a,b in zip(X, Y):   #柱子上的数字显示
        plt.text(a, b, '%.2f'%b, ha='center', va='bottom', fontsize=7)
    plt.title(f"Channel Theme", y=-0.11)
    plt.show()


# `is` 类的分类统计
def day_stat(data):
    day_count = []
    
    for name in [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
    ]:
        idx = find_index(data["col_names"], name)
        _data = np.array(data["data"][:, idx], dtype=int)
        day_count.append(sum(_data))
    print(day_count)
    plt.plot()
    X = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    Y = day_count
    plt.bar(x=X, height=Y)
    for a,b in zip(X, Y):   #柱子上的数字显示
        plt.text(a, b, '%.2f'%b, ha='center', va='bottom', fontsize=7)
    plt.title(f"Day Published", y=-0.11)
    plt.show()


if __name__ == '__main__':
    data_path = "./data"
    train_data = json.load(open(os.path.join(data_path, "train.json"), "r"))

    train_data["data"] = np.array(train_data["data"])
    train_data["label"] = np.array(train_data["label"])

    train_data = preprocess(train_data)

    print("feature names: ", train_data["col_names"])
    print("num train samples: ", len(train_data["data"][:]))
    print("num positive samples: ", sum(train_data["label"]))
    print("num negative samples: ", len(train_data["data"][:])-sum(train_data["label"]))

    sns.set()

    # feature_distribution(train_data)

    # feature_corr(train_data)
    # feature_label_corr(train_data)
    theme_stat(train_data)
    # day_stat(train_data)
