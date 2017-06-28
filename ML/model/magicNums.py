def magicNums(train):
    rate_2015_q2 = 1
    rate_2015_q1 = rate_2015_q2 / 0.9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
    rate_2012_q4 = rate_2013_q1 / 0.9832  #     maybe use 2013q1 as a base quarter and get rid of mult?
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011


    # train 2015
    train['average_q_price'] = 1

    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


    # train 2014
    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


    # train 2013
    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


    # train 2012
    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


    # train 2011
    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

    train['price_doc'] = train['price_doc'] * train['average_q_price']


    #########################################################################################################

    mult = 1.054880504
    train['price_doc'] = train['price_doc'] * mult
    y_train = train["price_doc"]

    #########################################################################################################

    train = train.drop(["average_q_price"], axis=1)
    return train
