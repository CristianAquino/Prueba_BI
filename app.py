import streamlit as st
from multiapp import MultiApp
from apps import forestpractice # import your app modules here
from apps import SVR

from apps import RLogistica

from apps import KNN
from apps import LSMT
from apps import SVC

app = MultiApp()

app.add_app("RANDOM FOREST CLASSIFIER", forestpractice.app)
app.add_app("SUPPORT VECTOR REGRESION (SVR)", SVR.app)

app.add_app("REGRESION LOGISTICA", RLogistica.app)
app.add_app("KNN", KNN.app)
app.add_app("LONG-SHORT TERM MEMORY (LSMT)", LSMT.app)
app.add_app("SUPPORT VECTOR CLASSIFIER (SVC)", SVC.app)

# The main app
app.run()
