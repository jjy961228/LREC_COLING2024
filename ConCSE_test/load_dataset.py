import pandas as pd
import ipdb

class LoadNLI:
    @staticmethod
    def en_train():
        train_path = '../Koglish_dataset/nli/nli_cs_check/parsed_train.csv' 
        train_data = pd.read_csv(train_path, sep=',')
        return train_data 
    @staticmethod
    def cross_train():
        train_path = '../Koglish_dataset/nli/nli_cs_check/cross_train.csv' 
        train_data = pd.read_csv(train_path, sep=',')
        return train_data
    
class LoadSTSB:
    @staticmethod
    def en_valid():
        valid_path = '../Koglish_dataset/stsb_for_ConCSE/parsed_valid_ConCSE.csv' 
        valid_data = pd.read_csv(valid_path, sep=',')
        return valid_data
    def en_test():
        test_path = '../Koglish_dataset/stsb_for_ConCSE/parsed_valid_ConCSE.csv' 
        test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    @staticmethod
    def cross_valid():
        valid_path = '../Koglish_dataset/stsb_for_ConCSE/cross_valid_ConCSE.csv' 
        valid_data = pd.read_csv(valid_path, sep=',')
        return valid_data
    def cross_test():
        test_path = '../Koglish_dataset/stsb_for_ConCSE/cross_test_ConCSE.csv'  
        test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
class TestLoadSTS_12:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts12/sts12_parsed/parsed_test.csv' 
        test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    def cross_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts12/sts12_cross/cross_test.csv' 
        test_data = pd.read_csv(test_path, sep=',')
        return test_data

class TestLoadSTS_13:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts13/sts13_parsed/parsed_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    def cross_test(self):
        if self.eval_type == 'transfer' :
            test_path = '../Koglish_dataset/sts_12_16/sts13/sts13_cross/cross_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data

class TestLoadSTS_14:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts14/sts14_parsed/parsed_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data
    def cross_test(self):
        if self.eval_type == 'transfer' :
            test_path = '../Koglish_dataset/sts_12_16/sts14/sts14_cross/cross_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data

class TestLoadSTS_15:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts15/sts15_parsed/parsed_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    def cross_test(self):
        if self.eval_type == 'transfer' :
            test_path = '../Koglish_dataset/sts_12_16/sts15/sts15_cross/cross_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data

class TestLoadSTS_16:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sts_12_16/sts16/sts16_parsed/parsed_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    def cross_test(self):
        if self.eval_type == 'transfer' :
            test_path = '../Koglish_dataset/sts_12_16/sts16/sts16_cross/cross_test.csv' 
            test_data = pd.read_csv(test_path, sep=',')
        return test_data

class TestLoadSICK:
    def __init__(self, args):
        self.eval_type = args.eval_type
    def en_test(self):
        if self.eval_type == 'transfer':
            test_path = '../Koglish_dataset/sick/sick_parsed/parsed_test.csv' 
        test_data = pd.read_csv(test_path, sep=',')
        return test_data
    
    def cross_test(self):
        if self.eval_type == 'transfer' :
            test_path = '../Koglish_dataset/sick/sick_cross/cross_test.csv' 
        test_data = pd.read_csv(test_path, sep=',')
        return test_data

    