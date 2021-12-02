import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


### Loading Accuracy Data
acc_false_google = pd.read_csv('./ocr-batch_google_False_scores/AVERAGES.csv') #False (Google)
acc_false_aws = pd.read_csv('./ocr-batch_aws_False_scores/AVERAGES.csv')  #False (AWS)
acc_31_google = pd.read_csv('./ocr-batch_google_True_31_scores/AVERAGES.csv')  #True 31 (Google)
acc_31_aws = pd.read_csv('./ocr-batch_aws_True_31_scores/AVERAGES.csv')     #True 31 (AWS)


### Graph A
false_google_sort = acc_false_google['average_score'].argsort()
false_aws_sort = acc_false_aws['average_score'].argsort()
plt.figure()
false_google_plot = plt.plot(acc_false_google['average_score'][false_google_sort].values, label='No Preprocessing (Google)')
false_aws_plot = plt.plot(acc_false_aws['average_score'][false_aws_sort].values, label='No Preprocessing (AWS)')
true_google_plot = plt.plot(acc_31_google['average_score'][false_google_sort].values, label='Window Size 31 (Google)')
true_aws_plot = plt.plot(acc_31_aws['average_score'][false_aws_sort].values, label='Window Size 31 (AWS)')
plt.legend(bbox_to_anchor=(0,1), loc='upper left', borderaxespad=0)
plt.xlabel('Image Index (Sorted based on No Preprocessing Curve)')
plt.ylabel('Recognition Accuracy (%)')
plt.title('Average Text Recognition Accuracy over all Images for each Configuration')
plt.show()


acc_false_google['Preprocessing']= 'No Preprocessing (Google)'
acc_false_aws['Preprocessing']= 'No Preprocessing (AWS)'
acc_31_google['Preprocessing']= 'Window Size 31 (Google)'
acc_31_aws['Preprocessing']= 'Window Size 31 (AWS)'
accuracies = pd.concat([acc_false_google, acc_false_aws, acc_31_google, acc_31_aws])




### Graph B

sns.boxplot(x='Preprocessing',y='average_score',data=accuracies).set(xlabel='Configuration',ylabel='Average Accuracy Score (%)', title='Average Accuracy Scores by Configuration')
plt.show()

### Loading Timing Data
time_false_google = pd.read_csv('./result_google/timings_google_False.csv')
time_false_google['process']=time_false_google['process'].fillna(0)
time_31_google = pd.read_csv('./result_google/timings_google_True_31.csv')

time_false_aws = pd.read_csv('./timings_aws/timings_ocr-batch_aws_False.csv')
time_false_aws['process']=time_false_aws['process'].fillna(0)
time_31_aws = pd.read_csv('./timings_aws/timings_ocr-batch_aws_True_31.csv')


### Graph D
mean_df = pd.DataFrame(columns=['load','process','detect'], index=['No Preprocessing (Google)','No Preprocessing (AWS)','Window Size 31 (Google)','Window Size 31 (AWS)'])

mean_df.loc['No Preprocessing (Google)',:]=time_false_google[['load', 'process', 'detect']].mean()
mean_df.loc['No Preprocessing (AWS)',:]=time_false_aws[['load','process','detect']].mean()
mean_df.loc['Window Size 31 (Google)',:]=time_31_google[['load', 'process', 'detect']].mean()
mean_df.loc['Window Size 31 (AWS)',:]=time_31_aws[['load','process','detect']].mean()
mean_df = mean_df.rename(columns={'load': 'Loading Image', 'process': 'Preprocessing Image', 'detect': 'Text Detection'})

ax = mean_df.plot.bar(rot=0,xlabel='Configuration',ylabel='Time (seconds)',title='Average Time to Complete each Step Across All Images for Each Configuration')
plt.show()
