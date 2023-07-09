import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression as mir
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb
import lightgbm as lgbm
from itertools import product
from scipy import signal
from scipy import stats
from statsmodels.tsa.deterministic import Fourier

COUNTRIES = ['DE', 'FR']

weather_vars = ['DE_TEMP', 'FR_TEMP', 'DE_RAIN', 'FR_RAIN', 'DE_WIND', 'FR_WIND']

def fill_weather_gap(data, de_rain, de_wind, de_temp, fr_rain, fr_wind, fr_temp):
	final = data.copy()
	df = data.copy()
	models = [de_temp, fr_temp, de_rain, fr_rain, de_wind, fr_wind]
	fill_idx = df[df['DE_WIND'].isna()].index
	left_idx = df[df.index < fill_idx.min()].index
	right_idx = df[df.index > fill_idx.max()].index
	n = fill_idx.size

	offsets = {}

	for w in weather_vars:
		df[w] = df[w] + abs(df[w].min()) + 1
		offsets[w] = abs(df[w].min()) + 1

	extra_fourier = [Fourier(period=i, order=3) for i in [365.25]]
	dp = DeterministicProcess(
	    constant=True,
	    period=52,
	    index=df.index,
	    order=1,
	    fourier=3,
	    additional_terms=extra_fourier
	)

	df = pd.concat([df, dp.in_sample()], axis=1)

	# fit left side

	X = df.copy()
	lag_amt = 3
	X = pd.concat([X, lag_shift(X[weather_vars], range(1, lag_amt), only_shifts=True)], axis=1).drop(['DAY_ID', 'COUNTRY'], axis=1)

	# TODO: induce lags, features

	for i in range(n):
		for j in range(6):
			curr_idx = fill_idx.min() + i
			print(f'predicting row {curr_idx}, column {weather_vars[j]}...')
			models[j].fit(X.iloc[:(curr_idx)].drop(weather_vars[j:], axis=1), X.iloc[:(curr_idx)][weather_vars[j]])
			result = models[j].predict(np.array(X.iloc[(curr_idx)].drop(weather_vars[j:])).reshape(1, -1))
			##### make sure to delete the line below when implementing interpolation #####
			df.iat[curr_idx, df.columns.get_loc(weather_vars[j])] = result
			X.iat[curr_idx, X.columns.get_loc(weather_vars[j])] = result
			for k in range(1, lag_amt):
				X.iat[curr_idx + k, X.columns.get_loc(f"{weather_vars[j]}_SHIFT_{k}")] = result
			if curr_idx in range(472, 477):
				print(models[j].coef_)
	return df

def make_cum_prices(data):
	return

def time_series_test(tss, model, x, y, extra=None, wind_excess=False, graph_residuals=False, method='mape'):
	for (train, test) in tss.split(x):
		if wind_excess:
			df = make_wind_excess(x, train)
		else:
			df = x.copy()
		model.fit(df.iloc[train], y.iloc[train])
		train_output, test_output = model.predict(df.iloc[train]), model.predict(df.iloc[test])
		print(f'mape test: {mape(test_output, y.iloc[test])}')
		print(f'mape train: {mape(train_output, y.iloc[train])}')
		print(f'mse test: {mse(test_soutput, y.iloc[test])}')
		print(f'mse train: {mse(train_output, y.iloc[train])}')
		plt.figure()
		after_idx = y.loc[~y.index.isin(train)].index.intersection(y.loc[~y.index.isin(test)].index)
		fig, ax = plt.subplots(figsize=(16, 6))
		sns.lineplot(x=after_idx, y=y.iloc[after_idx], color='red')
		sns.lineplot(x=train, y=y.iloc[train], color='blue')
		# sns.lineplot(x=test, y=y.iloc[test], color='blue')
		sns.lineplot(x=test, y=test_output, color='orange')
		if graph_residuals:
			plt.figure()
			sns.lineplot(x=test, y=test_output - y.iloc[test], color='red')
		plt.show()

def make_features(data):
	df = data.copy()
	df = basic_clean(df)

# training metric
def metric_train(output, test, method='spearman'):
	if method == 'mape':
		print(f'mape score: {mape(output, test)}')
		return 0
	return spearmanr(output, test).correlation

# call whenever testing models
def test_model(model_0, x_train_0, x_test_0, y_train, y_test, model_1=None, x_train_1=None, x_test_1=None, detailed=False, print_output=False, graph_residuals=False, method='spearman'):
	model_0.fit(x_train_0, y_train)
	train_output = model_0.predict(x_train_0)
	test_output = model_0.predict(x_test_0)

	if model_1 is not None:
		if detailed:
			print('model_0 fit on test set: {:.1f}%'.format(100 * metric_train(test_output, y_test)))
			print('model_0 fit on train set: {:.1f}%'.format(100 * metric_train(train_output, y_train)))
		if x_train_1 is None:
			x_train_1 = x_train_0
		if x_test_1 is None:
			x_test_1 = x_test_0
		train_residuals = train_output - y_train
		test_residuals = test_output - y_test
		model_1.fit(x_train_1, train_residuals)
		output_train_residual = model_1.predict(x_train_1)
		output_test_residual = model_1.predict(x_test_1)
		if graph_residuals:
			plt.clf()
			plt.figure()
			fig, ax = plt.subplots(2)
			p = sns.lineplot(x=y_test.sort_index().index, y=test_residuals.sort_index(), ax=ax[0])
			ax[0].set_xlabel('MODEL 0 TEST RESIDUALS')
			plt.figure()
			p = sns.lineplot(x=y_train.sort_index().index, y=train_residuals.sort_index(), ax=ax[1])
			ax[1].set_xlabel('MODEL 0 TRAIN RESIDUALS')
		if detailed:
			print('model_1 fit on test residuals: {:.1f}%'.format(100 * metric_train(output_test_residual, test_residuals, method=method)))
			print('model_1 fit on train residuals: {:.1f}%'.format(100 * metric_train(output_train_residual, train_residuals, method=method)))
		train_output = train_output - output_train_residual
		test_output = test_output - output_test_residual

	print('fit on test set: {:.1f}%'.format(100 * metric_train(test_output, y_test)))
	print('fit on training set: {:.1f}%'.format(100 * metric_train(train_output, y_train)))
	print('')
	if print_output:
		print(test_output)
	if graph_residuals:
		plt.clf()
		plt.figure()
		p = sns.lineplot(x=y_test.sort_index().index, y=(test_output - y_test).sort_index())
		p.set(xlabel='TEST RESIDUALS')
		plt.figure()
		p = sns.lineplot(x=y_train.sort_index().index, y=(train_output - y_train).sort_index())
		p.set(xlabel='TRAIN RESIDUALS')
		plt.show()
	return train_output, test_output

def kf_test_model(kf, model, x, y, extra=None, wind_excess=True, target_col='TARGET', graph_residuals=False, method='spearman'):
	for (train, test) in kf.split(x):
		if wind_excess:
			df = make_wind_excess(x, train)
		else:
			df = x.copy()
		if target_col:
			test_model(model, df.iloc[train], df.iloc[test], y.iloc[train][target_col], y.iloc[test][target_col], model_1=extra, method=method)
		else:
			test_model(model, df.iloc[train], df.iloc[test], y.iloc[train], y.iloc[test], model_1=extra, graph_residuals=graph_residuals, method=method)

# make sure dataframe is sorted first!

def lag_shift(data, steps=[1], only_shifts=False):
	df = data.copy()
	out = df.copy()
	if only_shifts:
		out = pd.DataFrame(index=df.index)
	for step in steps:
		df_shifted = df.shift(step, fill_value=0).add_suffix(f'_SHIFT_{step}')
		out = pd.concat([out, df_shifted], axis=1)
	return out


# fills nan weather values by

# fills nan exchange values by 

# fills nan import/export (note: only france has missing exchange and imp/exp values, hence the much shittier predictions!)

# fill nan values with median, drop day_id
def basic_clean(data):
	df = data.copy().drop('DAY_ID', axis=1)
	return df.fillna(df.median(numeric_only=True))

def drop_clean(data):
	df = data.copy().drop('DAY_ID', axis=1)
	return df.dropna()

# factorize country column
def enum_country(data):
	df = data.copy()
	df['COUNTRY'] = df['COUNTRY'].factorize()[0]
	return df

# transforms wind forecasts into right dimensions (reasoning is that wind forecasts would represent net flow of volume, while
# wind turbines roughly generate energy based on area, so we cube root wind and then square it)
def make_wind_sqcb(data, drop_wind=True):
	df = data.copy()
	df['DE_WIND_SQCB'] = (df['DE_WIND'] - df['DE_WIND'].min()).pow(2.0/3.0)
	df['FR_WIND_SQCB'] = (df['FR_WIND'] - df['FR_WIND'].min()).pow(2.0/3.0)
	if drop_wind:
		df = df.drop(['DE_WIND', 'FR_WIND'], axis=1)
	return df

# determines over- or underproduction of wind power based on forecasts
def make_wind_excess(data, train_idx, wind='WIND_SQCB', de_threshold=1.5, fr_threshold=1.5, drop_windpow=True, use_iloc=True):
	df = data.copy()
	x = df.iloc[train_idx] if use_iloc else df.loc[train_idx]
	lr = SDLinReg()
	lr.fit(x.copy(), 'DE_' + wind, 'DE_WINDPOW', lambda x, y : x > de_threshold)
	df['DE_WIND_EXCESS'] = lr.predict(df)
	lr.fit(x.copy(), 'FR_' + wind, 'FR_WINDPOW', lambda x, y : x > fr_threshold)
	df['FR_WIND_EXCESS'] = lr.predict(df)
	if drop_windpow:
		df = df.drop(['DE_WINDPOW', 'FR_WINDPOW'], axis=1)
	return df


# pretty sure the country column represents the country whose electricity future we're looking at, so we have to flip the sign of some things,
# and take import/export based on country—make sure to use this before one hot encoding country / converting to 0/1

def country_flow(data):
	df = data.copy()
	df['EXCHANGE'] = np.where(df['COUNTRY'] == 'DE', df['DE_FR_EXCHANGE'], df['FR_DE_EXCHANGE'])
	df['SELF_EXPORT'] = np.where(df['COUNTRY'] == 'DE', df['DE_NET_EXPORT'], df['FR_NET_EXPORT'])
	df['OTHER_EXPORT'] = np.where(df['COUNTRY'] == 'DE', df['FR_NET_EXPORT'], df['DE_NET_EXPORT'])
	df['SELF_CONSUMPTION'] = np.where(df['COUNTRY'] == 'DE', df['DE_CONSUMPTION'], df['FR_CONSUMPTION'])
	df['OTHER_CONSUMPTION'] = np.where(df['COUNTRY'] == 'DE', df['FR_CONSUMPTION'], df['DE_CONSUMPTION'])
	df = df.drop(['DE_CONSUMPTION', 'FR_CONSUMPTION', 'DE_FR_EXCHANGE', 'FR_DE_EXCHANGE', 'DE_NET_EXPORT', 'DE_NET_IMPORT', 'FR_NET_EXPORT', 'FR_NET_IMPORT'], axis=1)
	return df

def normalize(data):
	df = data.copy()
	return (df - df.mean()) / df.std()

def normalize_ret(data):
	df = data.copy()
	norm_cols = ['GAS_RET', 'COAL_RET', 'CARBON_RET']
	return (d[norm_cols] - d[norm_cols].mean()) / d[norm_cols].std()

def fuel_cost(data):
	df = data.copy()
	ls = ['GAS', 'COAL']
	for country, fuel in product(COUNTRIES, ls):
		df[f'{country}_{fuel}_COST'] = df[f'{country}_{fuel}'].multiply(df[f'{fuel}_RET'])
		df[f'{country}_{fuel}_CARBON'] = df[f'{country}_{fuel}'].multiply(df['CARBON_RET'])
	df['DE_LIGNITE_CARBON'] = df[f'DE_LIGNITE'].multiply(df['CARBON_RET'])
	return df

def fourier_features(data, freq=[365, 106.81647765176784], order=3, include_time=True):
	df = data.copy()
	time = df.index
	for fq in freq:
		k = 2 * np.pi * (1 / fq) * time
		for i in range(1, order + 1):
			df[f"SIN_{fq}_{i}"] = np.sin(i * k)
			df[f"COS{fq}_{i}"] = np.cos(i * k)
	df['TIME'] = df.index
	return df

# for fitting two part linear regression to WIND_SQCB / WINDPOW to determine excess production
# in general, given two series and a threshold (boolean) function, the fn will split the the x and y values
# based on the function and then do ols on each side, returning a new series with the residual
# of each y-value from the fitted line

def lr_sd(lr, x, y):
	return (y - lr.predict(x)).pow(2).sum() / y.size
  
class SDLinReg:
	def __init__(self):
		return None
	
	def fit(self, d, x, y, f=None):
		self.x = x
		self.y = y
		data = d[[x, y]].copy()
		if f is None:
			d1 = data
			lr1 = LinearRegression()
			lr1.fit(d1[x].values.reshape(-1, 1), d1[y])
			sd1 = lr_sd(lr1, d1[[x]], d1[y])
			self.p = lambda r : (r[1] - lr1.predict([[r[0]]])) / sd1
		else:
			d1, d2 = data[f(data[x], data[y])], data[~ f(data[x], data[y])]
			lr1, lr2 = LinearRegression(), LinearRegression()
			lr1.fit(d1[x].values.reshape(-1, 1), d1[y])
			lr2.fit(d2[x].values.reshape(-1, 1), d2[y])
			sd1, sd2 = lr_sd(lr1, d1[[x]], d1[y]), lr_sd(lr2, d2[[x]], d2[y])
			self.p = lambda r : ((r[1] - lr1.predict([[r[0]]])) / sd1) if f(r[0], r[1]) else ((r[1] - lr2.predict([[r[0]]])) / sd2)
		return

	def predict(self, d, x=None, y=None, debug=False):
		if x is None:
			x = self.x
		if y is None:
			y = self.y
		data = d[[x, y]]
		if debug:
			i = 0
			for row in data.itertuples(index=False):
				print("row:", row)
				print("p(row):", self.p(row))
				print("float:", float(self.p(row)))
				i += 1
				if i > 20:
					break
		return pd.Series([float(self.p(row)) for row in data.itertuples(index=False)], index=d.index)