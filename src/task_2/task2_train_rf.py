import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report


root_path = "/Users/kinjal/Desktop/Spring2020/11797/QA_SDP/src/dataloaders/"
dataset = pd.read_csv("%straindata.csv" % root_path, header=None)


dataset.columns = ['cite_id', 'ref_id', 'cite_line_ratio', 'ref_line_ratio', 'isPercentPresent',
                   'isFloatingPointPresent', 'facet_prob_aim', 'facet_prob_hypothesis', 'facet_prob_implication',
                   'facet_prob_method', 'facet_prob_result', 'facet_section_prob_aim', 'facet_section_prob_hypothesis',
                   'facet_section_prob_implication', 'facet_section_prob_method', 'facet_section_prob_result',
                   'is_aimcitation', 'is_hypothesiscitation', 'is_implicationcitation', 'is_methodcitation',
                   'is_resultcitation']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

train, test = train_test_split(dataset, test_size=0.2)
x_train = train.iloc[:, 2:16]
y_train = train.iloc[:, 16:]
x_test = test.iloc[:, 2:16]
y_test = [test.iloc[i, 16:].tolist() for i in range(0, len(x_test))]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rfc = RandomForestClassifier(bootstrap=True, max_depth=10, max_features='sqrt', random_state=1)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

# print(len(y_pred), y_pred)
# print(len(y_test), y_test)

num_correct = (y_pred == y_test).sum().item()
print("Test Accuracy : ", num_correct / len(x_test))
print("Classification Report: ", classification_report(y_test, y_pred))
F1metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('MACRO F1score:', F1metrics[2])
F1metrics = precision_recall_fscore_support(y_test, y_pred, average='micro')
print('MICRO F1score:', F1metrics[2])
