import streamlit as st
import pandas as pd
import numpy as np
import lazypredict
from sklearn.model_selection import train_test_split

st.markdown("<center><h1> AUTOML <small>by I. A. ITALIA</small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" >Non <bold>sei stanco di dover provare decine di modelli solo per capire il più accurato</bold> per i tuoi dati <bold>  ?</bold><p>', unsafe_allow_html=True)

dataframe = pd.DataFrame()

file_caricato =  st.file_uploader("SCEGLI UN FILE CSV", type="CSV", accept_multiple_files=False)
if file_caricato is not None:
	dataframe = pd.read_csv(file_caricato)
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	dataframe = dataframe.select_dtypes(include=numerics)
	dataframe = dataframe.dropna()
	with st.beta_expander("VISUALIZZA DATASET"):
		st.write(dataframe)
	with st.beta_expander("STATISICA DI BADE"):
		st.write(dataframe.describe())
	
	st.markdown("<br><br>", unsafe_allow_html=True)	
	colonne = dataframe.columns
	target = st.selectbox('Scegli la variabile Target', colonne )
	st.write("target impostato su " + str(target))
	colonne = colonne.drop(target)
	descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
	st.write("Variabili Indipendenti impostate su  " + str(descrittori))
	
	
	problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
	tipo_di_problema = st.selectbox('Seleziona uno o più argomenti per le ultime notizie', problemi)
	percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
	
	X = dataframe[descrittori]
	y = dataframe[target]
	
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)

	if(st.button("DIMMI QUALE E' IL MIGLIO MODELLO PER CLASSIFICARE/STIMARE IL MIO TARGET !")):
		if(tipo_di_problema == "CLASSIFICAZIONE"):
			from lazypredict.Supervised import LazyClassifier
			clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
			models,predictions = clf.fit(X_train, X_test, y_train, y_test)
			st.write(models)
		if(tipo_di_problema == "REGRESSIONE"):
			from lazypredict.Supervised import LazyRegressor
			reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
			models, predictions = reg.fit(X_train, X_test, y_train, y_test)
			st.write(models)
