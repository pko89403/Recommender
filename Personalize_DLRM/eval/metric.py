import sklearn.metrics
from sklearn.metrics import auc
import numpy as np
import torch
from termcolor import colored
import sys
import warnings
import json

class eval(object):

    def __init__(self):

        with open('config/item_info.json') as json_file:
            self.item_info = json.load(json_file)

        with open('config/client_info.json') as json_file:
            self.client_info = json.load(json_file)

    def warn(*args, **kwargs):
        pass

    def metrics(self, total_iter, Error, y_pred, y_true, writer):

        try:
            y_pred = [1 if i > 0.4 else 0 for i in y_pred.detach().cpu().numpy()]

            with torch.no_grad():

                y_pred = np.array(y_pred)
                y_true = np.array(y_true)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
                
                print(colored(f"fpr : {fpr}", "red"))
                print(colored(f"tpr : {tpr}", "red"))
                print(colored(f"thresholds : {thresholds}", "red"))                                

                print("total_iter", colored(total_iter, 'blue'))
                print("auc", colored(auc(fpr, tpr), 'yellow'))
                print("recall", colored(sklearn.metrics.recall_score(y_true, y_pred), 'yellow'))
                print("precision", colored(sklearn.metrics.precision_score(y_true, y_pred), 'yellow'))
                print("error rate", colored(float(Error), 'yellow'), "\n")

                writer.add_scalars('metric', {"recall": sklearn.metrics.recall_score(y_true, y_pred),
                                             "auc": auc(fpr, tpr),
                                             "precision":sklearn.metrics.precision_score(y_true, y_pred),
                                             "Error" :float(Error)}, total_iter)

                return Error

        except Exception as e:
            print("metrics", e)

    def add_grph(self, recsys, writer):

        writer.add_graph()

    def close(self, writer):

        writer.close()

    def add_emb(self, recsys, writer):

        for index, emb in enumerate(recsys.model.emb_l):

            if recsys.emb_l_colName[index] == "prd_cd_cd":

                prd_nm = list()
                for i in range(len(emb.weight)):
                    prd_nm.append(self.item_info[str(i)]['prd_nm'])

                writer.add_embedding(emb.weight, metadata=prd_nm, global_step=1, tag=f'emb_l/{recsys.emb_l_colName[index]}')

            else:

                pass
                # writer.add_embedding(emb.weight, metadata=None, global_step=1,
                #                      tag=f'emb_l/{recsys.emb_l_colName[index]}')