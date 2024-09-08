import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,log_loss
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from scipy.spatial import distance


#returns dataframe, candidates
def clean():
      #reading data:
      np.set_printoptions(suppress=True) #no scientific notation 
      df = pd.read_csv("KeplerPlanets.csv")
      #Dropping: Row ID, Flags, Error Bounds
      colsToDrop = ['rowid', 'koi_score','kepid', 'kepler_name', 'koi_pdisposition',
             "koi_fpflag_nt","koi_fpflag_ss", 'koi_fpflag_co',  
             'koi_period_err1', 'koi_period_err2','koi_time0bk_err1',
             'koi_time0bk_err2', 'koi_impact_err1', 'koi_impact_err2',
              'koi_duration_err1', 'koi_duration_err2', 
             'koi_depth_err1', 'koi_depth_err2',  'koi_prad_err1',
             'koi_prad_err2', 'koi_teq_err1', 'koi_teq_err2', 
             'koi_insol_err1', 'koi_insol_err2','koi_tce_plnt_num',
             'koi_tce_delivname','koi_steff_err1', 'koi_steff_err2',
             'koi_slogg_err1', 'koi_slogg_err2', 
             'koi_srad_err1', 'koi_srad_err2']
      colsToDrop2 = ['rowid', 'koi_score','kepid', 'kepler_name', 'koi_pdisposition',
             'koi_teq_err1', 'koi_teq_err2', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',  
             'koi_tce_plnt_num','koi_tce_delivname',]

      #cleaning data:
      df = df.drop(columns=colsToDrop) #drop uneeded columns, choose cols or cols2 to choose using error or not
      df = df.dropna(axis=0) #drop empty rows 

      #create list of candidates to use as test
      candidates = df[df['koi_disposition']== 'CANDIDATE']
      candidatelabels = candidates['kepoi_name'] #lables 
      candidates = candidates.drop(columns = 'kepoi_name') #drop labels 
      

      df = df[df['koi_disposition'] != 'CANDIDATE'] #remove candidate rows 
      
      df = df.replace("FALSE POSITIVE", 0) #change to numerical values for target
      df = df.replace("CONFIRMED", 1)
      df = df.sample(frac=1).reset_index(drop=True) #shuffle
       #add ,random_state=? to seed 

      dflabels = df['kepoi_name']#list of labels for data 
      df = df.drop(columns = 'kepoi_name')
     
      return [df,candidates,candidatelabels, dflabels]

temp = clean()
df = temp[0]
candidates = temp[1]
candidatelabels = temp[2]
dflabels = temp[3]

#print((len(df[df["koi_disposition"] == 0]))/len(df))
#CHECKING PERCENTAGES OF FALSE POSITIVES 
#FALSE POSITIVES MAKE UP 63% OF DATA - GOOD FOR FSCORE

#splitting data:
train = df[:round(df.shape[0]*.8)] #make sets for train and test 
test = df[round(df.shape[0]*.8):]

#make sets for candidates:
candidates = candidates.drop(columns = ['koi_disposition','koi_fpflag_ec'])

targetTrain = train['koi_disposition'] #train and test sets 
targetTest = test['koi_disposition']
FlagTrainTarget = train[ 'koi_fpflag_ec'] #flag target sets, alter string to choose flag
FlagTestTarget = test[ 'koi_fpflag_ec']

train = train.drop(columns = ['koi_disposition',  'koi_fpflag_ec']) #drop the target and flag out of the data 
test = test.drop(columns = ['koi_disposition',  'koi_fpflag_ec'])

train = np.array(train) #turn all into numpy arrays 
test = np.array(test)
targetTrain = np.array(targetTrain) 
targetTest = np.array(targetTest)
candidates = np.array(candidates)


scaler = StandardScaler() 
train = scaler.fit_transform(train) #standardize data 
test = scaler.transform(test)
candidates = scaler.transform(candidates)


def decisionTree():
	#predicting accuracy and fscore
	model = DecisionTreeClassifier(criterion='entropy') #create model
	model.fit(train,targetTrain) #fit data 
	predictedVals = model.predict(test) #grab predicted values 
	acc = model.score(test, targetTest) #score the model - accuracy 
	fscore = f1_score(targetTest, predictedVals, average='binary') #f1 score
	
	#checking a specific candidate planet's prediction 
	candidatesPred = model.predict(candidates)
	count = 0
	lis = list(candidatelabels)
	for i in range(len(lis)):
		if lis[i] == "K00799.01": #list specfic planet here
			print(candidatesPred[i])
		count+=1

	importance = permutation_importance(model, test, predictedVals)
	
	for i in range(len(testdf.columns)):
		print(testdf.columns[i])
		print(round(importance.importances_mean[i],4))

def logisticReg():
	#FINDS BEST C VALUE TO MINIMIZE LOG LOSS 
	temp = []
	arr= (np.arange(0,7,.1))
	for i in arr:
		model2 = LogisticRegression(C=10**i).fit(train, targetTrain)
		pred = model2.predict_proba(test)
		loss = log_loss(targetTest, pred)
		temp.append(loss)
	temp2 = min(temp)
	val = arr[temp.index(temp2)]

	#now creating model with found best C value - using on candidates 
	model = LogisticRegression(C=10**val).fit(train, targetTrain)
	pred = model.predict_proba(candidates)
	print(model.decision_function(candidates))
	print(pred)
	print(list(candidatelabels))
	loss = log_loss(targetTest, pred) 
	#FIRST COLUMN IS PROBABILITY OF NOT
	#BEING EXOPLANET, SECOND IS PROBABILITY OF BEING ONE 

def dis():

	count = 0
	
	for i in range(len(dflabels)):
		if list(dflabels)[i] == "K00172.02": #comparing to specific kepler IDS
			val = count
		count+=1
	print(val)
	distances = []
	for i in range(len(train)):
		distances.append(distance.euclidean(train[i],train[val]))
	distances.remove(0) #remove the smallest value of 0 - because the distance 
	#between a point and itself is 0. 
	temp = min(distances)
	val = dflabels[distances.index(temp)]
	print(val)

def falsePos():
	model = DecisionTreeClassifier(criterion='entropy')
	model.fit(train,FlagTrainTarget)
	pred = model.predict(test)
	importance = permutation_importance(model, test, pred)
	for i in range(len(testdf.columns)):
		print(testdf.columns[i])
		print(round(importance.importances_mean[i],4))
	
	acc = model.score(test, FlagTestTarget) #score the model - accuracy 
	fscore = f1_score(FlagTestTarget , pred, average='binary')
	print(acc)
	print(fscore)







