import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import operator
# #import seaborn as sns; sns.set()
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 10
# from sklearn.metrics import confusion_matrix

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
        self.smoothing = 1
        self.num_words_per_class = None
        self.grouped_class = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID]
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. So use smooth methods

        self.Pr_dict = {}
        for c in set(train_label):
            self.Pr_dict[c] = {}

        grouped_class_word = train_data.groupby(['classIdx', 'wordIdx'])
        self.num_words_per_class = grouped_class_word.sum()
        self.grouped_class = train_data.groupby(['classIdx']).sum()['count'].to_dict()
        
        
        for (cls, wrd), data in self.num_words_per_class.iterrows():
            self.Pr_dict[cls][wrd] = (data['count'] + self.smoothing) / (self.grouped_class[cls] + (self.smoothing * self.num_vocab))
            
        #     count = 0
        #     for r in d.to_dict('records'):
        #         if (count < 5):
        #             print(r)
        #             print(r['count'])
        #             count += 1
        #         self.Pr_dict[r['classIdx']][r['wordIdx']] = (r['count'] + self.smoothing) / (self.num_words_per_class[r['classIdx']] + (self.smoothing * self.num_vocab))

        #### takes too long
        # for c in set(train_label):
        #     per_class = train_data.loc[train_data['classIdx'] == c] # dataframe of only class 'key'
        #     total_word_count = per_class['wordIdx'].sum()
        #     for i in range(1, self.num_vocab+1):
        #         word_count = per_class.loc[per_class['wordIdx'] == i].sum()
        #         self.Pr_dict[c][i] = (word_count + smoothing) / (total_word_count + self.num_vocab)


        # ============================================================
        print("Training completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication to avoid underflow
                ### Remember to add prior probability, i.e. self.pi

                total = np.log(self.pi[classIdx])
                for wordIdx, count in new_dict[docIdx].items():
                    try:
                        total += count * np.log(self.Pr_dict[classIdx][wordIdx])
                    except:
                        total += count * np.log(self.smoothing / (self.grouped_class[classIdx] + self.num_vocab))
                score_dict[classIdx] = total

                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi
            
            self.pi[c] = train_label.count(c) / total

            # ============================================================
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))

        print("-"*80)