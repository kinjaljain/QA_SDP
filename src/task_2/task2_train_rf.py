import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report


root_path = "./"
train = pd.read_csv("%straindata.csv" % root_path, header=None)
test = pd.read_csv("%stestdata.csv" % root_path, header=None)

train.columns = ['cite_id', 'ref_id', 'cite_line_ratio', 'ref_line_ratio', 'isPercentPresent',
                   'isFloatingPointPresent', 'facet_prob_aim', 'facet_prob_hypothesis', 'facet_prob_implication',
                   'facet_prob_method', 'facet_prob_result', 'facet_section_prob_aim', 'facet_section_prob_hypothesis',
                   'facet_section_prob_implication', 'facet_section_prob_method', 'facet_section_prob_result',
                   'is_aimcitation', 'is_hypothesiscitation', 'is_implicationcitation', 'is_methodcitation',
                   'is_resultcitation']
test.columns = ['cite_id', 'ref_id', 'cite_line_ratio', 'ref_line_ratio', 'isPercentPresent',
                   'isFloatingPointPresent', 'facet_prob_aim', 'facet_prob_hypothesis', 'facet_prob_implication',
                   'facet_prob_method', 'facet_prob_result', 'facet_section_prob_aim', 'facet_section_prob_hypothesis',
                   'facet_section_prob_implication', 'facet_section_prob_method', 'facet_section_prob_result',
                   'is_aimcitation', 'is_hypothesiscitation', 'is_implicationcitation', 'is_methodcitation',
                   'is_resultcitation']
# print('Shape of the dataset: ' + str(dataset.shape))
# print(dataset.head())

# train, test = train_test_split(dataset, test_size=0.2)
ids_train = train.iloc[:, :2].to_numpy().tolist()

x_train = train.iloc[:, 2:16]
y_train = train.iloc[:, 16:]

ids_test = test.iloc[:, :2].to_numpy().tolist()

x_test = test.iloc[:, 2:16]
y_test = [test.iloc[i, 16:].tolist() for i in range(0, len(x_test))]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# rfc = RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=9, max_features='sqrt',
#                              random_state=1, class_weight="balanced")
                             # class_weight=[{0: 613/(613-51), 1: 613/51}, {0: 613/(613-9), 1: 613/9},
                             #               {0: 613/(613-39), 1: 613/39}, {0: 613/(613-426), 1: 613/426},
                             #               {0: 613/(613-88), 1: 613/88}])

# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
clf = OneVsRestClassifier(LogisticRegression(multi_class='ovr', max_iter=1000, solver='lbfgs'))
clf.fit(x_train, y_train)

#
# self.classifier.predict(test_data)
# rfc.fit(x_train, y_train)


# y_pred_train = rfc.predict(x_train)
y_pred_train = clf.predict(x_train)

y_train = [train.iloc[i, 16:].tolist() for i in range(0, len(x_train))]
# print((y_pred_train == y_train))
# num_correct = (y_pred_train == y_train)
# print(len(x_train))
# print("Test Accuracy : ", num_correct / len(x_train))
print("Classification Report: ", classification_report(y_train, y_pred_train))
F1metrics = precision_recall_fscore_support(y_train, y_pred_train, average='macro')
print('MACRO F1score:', F1metrics[2])
F1metrics = precision_recall_fscore_support(y_train, y_pred_train, average='micro')
print('MICRO F1score:', F1metrics[2])


# y_pred = rfc.predict(x_test)
y_pred = clf.predict(x_test)



# print(len(y_pred), y_pred)
# print(len(y_test), y_test)
# print(len(x_test))
# num_correct = (y_pred == y_test).sum().item()
# print("Test Accuracy : ", num_correct / len(x_test))


def make_pred_json(ids, y_pred):
    results_task2 = {}
    for i in range(len(ids)):
        cite, ref = ids[i]
        pred = y_pred[i]

        if (ref, cite) not in results_task2:
            results_task2[(ref, cite)] = [pred]
        else:
            current = results_task2[(ref, cite)]
            current.append(pred)
    return results_task2

results_train = make_pred_json(ids_train, y_pred_train)
results_test = make_pred_json(ids_test, y_pred)

results_task2 = {**results_train, **results_test}

import pickle
with open('results_task2.pkl', 'wb') as f:
    pickle.dump(results_task2, f)

import numpy as np
y_test.extend(y_train)
y_pred = y_pred.tolist()
y_pred_train = y_pred_train.tolist()
y_pred.extend(y_pred_train)

print("Classification Report: ", classification_report(y_test, y_pred))
F1metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('MACRO F1score:', F1metrics[2])
F1metrics = precision_recall_fscore_support(y_test, y_pred, average='micro')
print('MICRO F1score:', F1metrics[2])