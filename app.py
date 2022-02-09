
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import inchi
from rdkit import DataStructs

our_color_discrete_map={
                  "unknown": "rgba(180, 180, 180, 0.24)",
                  "low": "rgba(5, 192, 5, 0.74)",
                  "medium": "rgba(255, 112, 0, 0.74)",
                  "high": "rgba(224, 19, 19, 0.74)",
                  "watching": "blue"
                  }

def user_input_features(bees):
    pesticides = ["{} - {}".format(a_, b_) for a_, b_ in zip(bees.index.map(str).values , bees['name'].values)]
    pesticide_selection = st.sidebar.selectbox('Which pesticide would you like to know more about?',  pesticides)
    n_neighbours = st.sidebar.slider('Neighbours', 1, 10, 3)
    
    data = {'n_neighbours': n_neighbours,
            'pesticide_selection': pesticide_selection}
    features = pd.DataFrame(data, index=[0])
    return features

def give_nearest_n(proj_3d, point, k):
    n = proj_3d.shape[0]
    data = proj_3d.copy()

    dist_matrix = np.zeros((n, n))
    for i in range(1,n):
        dist = np.linalg.norm(data[:i,:]-data[i,:] , axis=1)
        for id, d in enumerate(dist):
            dist_matrix[i,id] = d

    dist_matrix = dist_matrix + dist_matrix.T

    nearest_n = np.zeros_like(dist_matrix, dtype=int)
    for id, row in enumerate(dist_matrix):
        nearest_n[id] = np.argsort(row)
    nearest_n = nearest_n[:,1:]
    return nearest_n[point, 0:k]

def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()+1))
    return mol

def get_labels(bees, slected_pesticide_idx):
    y_full = bees['honeybees_contact_kill_risk'].values
    y_full = y_full.add_categories('watching')
    y_full[slected_pesticide_idx] = 'watching' # What point do you want to watch?

    return y_full

def plot_2d(proj_2d, bees, slected_pesticide_idx):
  y_full = get_labels(bees, slected_pesticide_idx)

  fig_2d = px.scatter(
      proj_2d, x=0, y=1,
          hover_data=[bees.index.map(str).values, bees['name'].values],
          color=y_full, labels={'color': 'risk'},
          color_discrete_map=our_color_discrete_map,
  )
  return fig_2d

def plot_3d(proj_3d, bees, slected_pesticide_idx):
  y_full = get_labels(bees, slected_pesticide_idx)


  fig_3d = px.scatter_3d(
      proj_3d, x=0, y=1, z=2,
        hover_data=[bees.index.map(str).values, bees['name'].values],
          color=y_full, labels={'color': 'risk'},
          color_discrete_map=our_color_discrete_map,
  )
  fig_3d.update_traces(marker_size=5)

  fig_3d.show()

  return fig_3d


def streamlit_stuff():

  st.write("""
  # Bee-Friendly Pesticide Classifier

  This app lets you choose one pesticide from a collection of pesticides and predicts the risk of death for a bee if she encounters that specific pesticide.
  We infer the risk by looking at the risk-rating of similar pesticides (from a chemical molecule structure perspective).
  """)

  st.sidebar.header('User Input Parameters')

  bees = pd.read_pickle('scraped_molecules_honeybees_working.pickle')
  bees['honeybees_contact_kill_risk'] = bees['honeybees_contact_kill_risk'].cat.add_categories('unknown').fillna("unknown")
  df = user_input_features(bees)

  st.subheader('User Input Parameters')
  st.write(df)

  slected_pesticide = df['pesticide_selection'].values[0]
  slected_pesticide_idx = int(slected_pesticide.split("-")[0][:-1])
  st.write(slected_pesticide, slected_pesticide_idx)

  proj_2d = np.load("2317_proj_2d.npy")
  st.header("2D UMAP")
  st.plotly_chart(plot_2d(proj_2d, bees, slected_pesticide_idx))

  proj_3d = np.load("2317_proj_3d.npy")
  st.header("3D UMAP")
  st.plotly_chart(plot_3d(proj_3d, bees, slected_pesticide_idx))


  slected_pesticide_row = bees.iloc[slected_pesticide_idx, :]
  st.header("Info about " + slected_pesticide_row['name'])
  st.write(f"This one is currently classified as {str(slected_pesticide_row['honeybees_contact_kill_risk'])}.")

  m = inchi.MolFromInchi(slected_pesticide_row['inchi'])
  fig = Draw.MolToMPL(m)
  st.pyplot(fig)



  # K-NN
  st.subheader('Nearest Neighbours')

  k = int(df['n_neighbours'].values[0])
  nn = give_nearest_n(proj_3d, slected_pesticide_idx, k)
  st.write(nn)

  df_nn = bees.iloc[nn]
  st.write(df_nn)

  for n in nn:
      # st.write(df_nn.loc[n])
      st.subheader('Neighbour ' + df_nn.loc[n, 'name'] + ': ' + str(df_nn.loc[n, 'honeybees_contact_kill_risk']))
      nn_mol = inchi.MolFromInchi(df_nn.loc[n, 'inchi'])
      fig = Draw.MolToMPL(nn_mol)
      st.pyplot(fig)




  # st.subheader('Class labels and their corresponding index number')
  # st.write(iris.target_names)

  # st.subheader('Prediction')
  # st.write(iris.target_names[prediction])


  # st.subheader('Prediction Probability')
  # st.write(prediction_proba)

streamlit_stuff()