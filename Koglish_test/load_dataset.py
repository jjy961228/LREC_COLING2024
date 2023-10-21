import pandas as pd
import ipdb

class LoadSST2:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/sst2/sst2_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/sst2/sst2_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/sst2/sst2_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/sst2/sst2_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/sst2/sst2_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/sst2/sst2_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data, valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/sst2/sst2_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/sst2/sst2_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/sst2/sst2_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data, valid_data, test_data
    
class LoadMRPC:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/mrpc/mrpc_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
class LoadCOLA:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/cola/cola_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/cola/cola_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/cola/cola_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/cola/cola_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/cola/cola_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/cola/cola_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/cola/cola_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/cola/cola_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/cola/cola_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
class LoadRTE:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/rte/rte_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/rte/rte_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/rte/rte_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/rte/rte_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/rte/rte_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/rte/rte_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/rte/rte_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/rte/rte_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/rte/rte_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data

class LoadSTSB:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/stsb/stsb_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/stsb/stsb_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/stsb/stsb_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/stsb/stsb_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/stsb/stsb_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/stsb/stsb_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/stsb/stsb_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/stsb/stsb_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/stsb/stsb_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    
class LoadQNLI:
    @staticmethod
    def en2en():
        train_path = '../Koglish_dataset/qnli/qnli_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/qnli/qnli_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/qnli/qnli_train_test_split/parsed_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def en2cross():
        train_path = '../Koglish_dataset/qnli/qnli_train_test_split/parsed_train.csv'
        valid_path = '../Koglish_dataset/qnli/qnli_train_test_split/parsed_valid.csv'
        test_path = '../Koglish_dataset/qnli/qnli_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data
    @staticmethod
    def cross2cross():
        train_path = '../Koglish_dataset/qnli/qnli_train_test_split/cross_train.csv'
        valid_path = '../Koglish_dataset/qnli/qnli_train_test_split/cross_valid.csv'
        test_path = '../Koglish_dataset/qnli/qnli_train_test_split/cross_test.csv'
        train_data = pd.read_csv(train_path, sep=',')
        valid_data = pd.read_csv(valid_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        return train_data , valid_data, test_data


    