
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import matplotlib.pyplot as plt

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

def give_nearest_n(proj_3d, point):
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
    return nearest_n[point,:], dist_matrix[point,1:]

def classify_pesticide(contact_kill_risk_of_knn, distances):
    distances = np.array(distances)
    risk_dict = {'low': 1, 'medium': 2, 'high': 3}
    weights = (-distances+distances.sum())
    weights = weights/weights.sum()
    mean = 0
    for id, risk in enumerate(contact_kill_risk_of_knn):
        mean += risk_dict[risk] * weights[id]
    mean = math.ceil(mean)
    return [risk for risk, val in risk_dict.items() if val==mean][0]

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
  st.set_page_config(page_title='Bee-Friendly', layout="centered")
  st.write("""
  # Bee-Friendly Pesticide K-NN Classifier

  On the left side you can choose one pesticide from a collection of pesticedes and the number of K-NearestNeighbours you want to compare it to.
  The App then compares the molecular structure of all pesticides inside our collection and calculates the distance between each molecule pair (fingerprint from inchi).
  Then the App uses UMAP for dimension reduction in order to plot the 3D Map which shows the molecules relative to each other. Molecules with small distances have a similar structure.
  Below the 3D Plot you can see the Classification according to the KNN and you can also compare the KNN molecules with the chosen one.
  """)

  st.sidebar.header('User Input Parameters')

  bees = pd.read_pickle('scraped_molecules_honeybees_working.pickle')
  bees['honeybees_contact_kill_risk'] = bees['honeybees_contact_kill_risk'].cat.add_categories('unknown').fillna("unknown")
  df = user_input_features(bees)

#   st.subheader('User Input Parameters')
#   st.write(df)

  slected_pesticide = df['pesticide_selection'].values[0]
  slected_pesticide_idx = int(slected_pesticide.split("-")[0][:-1])
#   st.write(slected_pesticide, slected_pesticide_idx)

#   proj_2d = np.load("2317_proj_2d.npy")
#   st.header("2D UMAP")
#   st.plotly_chart(plot_2d(proj_2d, bees, slected_pesticide_idx))

  proj_3d = np.load("2317_proj_3d.npy")
  st.header("3D UMAP")
  st.write("""
  The chosen pesticide will be marked blue inside the graph so you can find it.\n
  You can also hover over the Data Points in order to find out the molecule id (hover_data_0) which can be typed in on the left side.
  """)
  st.plotly_chart(plot_3d(proj_3d, bees, slected_pesticide_idx))

  # K-NN
  st.header('Nearest Neighbours')
  st.write('k=1: classifies 0.6197% of the test_set correctly. (crit_fail:9, near_wrong:231)')
  st.write('k=2: classifies 0.6878% of the test_set correctly. (crit_fail:8, near_wrong:189)')
  st.write('k=3: classifies 0.6545% of the test_set correctly. (crit_fail:14, near_wrong:204)')
  st.write('k=4: classifies 0.6434% of the test_set correctly. (crit_fail:19, near_wrong:206)')
  st.write('k=5: classifies 0.6308% of the test_set correctly. (crit_fail:18, near_wrong:215)')
  st.write('k=6: classifies 0.6323% of the test_set correctly. (crit_fail:23, near_wrong:209)')
  st.write('k=7: classifies 0.6244% of the test_set correctly. (crit_fail:22, near_wrong:215)')
  st.write('k=8: classifies 0.6292% of the test_set correctly. (crit_fail:21, near_wrong:213)')
  st.write('k=9: classifies 0.6260% of the test_set correctly. (crit_fail:22, near_wrong:214)')
  st.write('k=10: classifies 0.6260% of the test_set correctly. (crit_fail:21, near_wrong:215)')

  k = int(df['n_neighbours'].values[0])
  nn,dist = give_nearest_n(proj_3d, slected_pesticide_idx)
  
  # Filter out all neighbours which have no known kill risk entry.
  knn_list = []
  dist_list = []
  df_nn = bees.iloc[nn]
  for id, n in enumerate(nn):
      if df_nn.loc[n, 'honeybees_contact_kill_risk'] != 'unknown':
          knn_list.append(n)
          dist_list.append(dist[id])
  
  # Get the K-Nearest-Neighbours
  nn = knn_list[0:k]
  dist = dist_list[0:k]
  
  df_nn = bees.iloc[nn]
  kill_risks = list(df_nn['honeybees_contact_kill_risk'].values)
  classification_kill_risk = classify_pesticide(kill_risks, dist)
  
  slected_pesticide_row = bees.iloc[slected_pesticide_idx, :]
  st.subheader("Info about the chosen pesticide" + slected_pesticide_row['name'])
  st.write(f"This ones kill risk is currently `{slected_pesticide_row['honeybees_contact_kill_risk']}` -> The Model predicts that its kill risk is: `{classification_kill_risk}`.")
  st.write("Here you can see the structure of the molecule:")

  m = inchi.MolFromInchi(slected_pesticide_row['inchi'])
  fig = Draw.MolToMPL(m)
  st.pyplot(fig)
  
  # Get K-Nearest-Neighbours out of Dataframe and print them
  column_titles = ['name', 'honeybees_contact_kill_risk', 'honeybees_contact_kill_value_clean', 'url', 'inchi']
  df_nn = df_nn.reindex(columns=column_titles)
  st.subheader("Those are the K-Nearest Neighbours")
  st.write(df_nn)
  
  # Draw the molecular shapes of the KNNs
  for n in nn:
      # st.write(df_nn.loc[n])
      st.subheader('Neighbour ' + df_nn.loc[n, 'name'] + ': ' + f"`{str(df_nn.loc[n, 'honeybees_contact_kill_risk'])}`")
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
