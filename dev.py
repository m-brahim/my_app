import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
import markdown
from streamlit_folium import st_folium
from streamlit_extras.metric_cards import style_metric_cards
from mitosheet.streamlit.v1 import spreadsheet
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit_option_menu import option_menu
import random
import duckdb
from openai import OpenAI
import time
import json
import os
from streamlit_elements import elements, mui, html
from streamlit_elements import dashboard
from streamlit_elements import editor
from st_mui_table import st_mui_table


#config du titre de la page
st.set_page_config("Suivi des ventes de la société", page_icon="", layout="wide")

#collecte des données
url = "Exemple - Hypermarché_Achats.csv"

#charger le fichier CSS
with open("style.css") as f:
    css_code = f.read()

st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

#modif sur colonne Ventes
df = pd.read_csv(url, delimiter=";")
df['Ventes'] = df['Ventes'].str.replace('[^\d]', '', regex=True)
df['Ventes'] = pd.to_numeric(df['Ventes'], errors='coerce', downcast='integer')

#ajout d'une colonne année et une colonne mois qui extrait l'année de la date de commande
df['Année'] = pd.to_datetime(df['Date de commande'], format='%d/%m/%Y').dt.year
df['Mois'] = pd.to_datetime(df['Date de commande'], format='%d/%m/%Y').dt.month_name()

df = df.sort_values(by=['Année', 'Mois'])

df = df.reset_index(drop=True)

#tri dans l'ordre des années
sorted_years = sorted(df['Année'].unique())
sorted_years_2 = sorted(df['Année'].unique())


#création de colonnes
col_title, col_logo = st.columns([3, 0.5])

#une colonne pour le titre & une pour les listes déroulantes

with st.sidebar:
    selected3 = option_menu("Menu", ["Accueil", "Import", "OpenAI", "Tâches", 'Tests', 'Elements'], 
    icons=['house', 'cloud-upload', 'lightbulb', 'list-task', 'gear', ''], 
    menu_icon="cast", default_index=0,
    styles={
        "container": {"border": "1px solid #CCC", "border-left": "0.5rem solid #000000", "border-radius": "5px", "box-shadow": "0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15)", "height" : "800px"},
        "icon": {"color": "black", "font-size": "16px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#f7e98e"},
        "nav-link-selected": {"background-color": "#f7e98e"},
        "nav-link.active" : {"color": "black"}
    }
)



















if selected3 == "Accueil" :
    #1) analyse client
    with col_title:
        st.title("Suivi des ventes de la société")
        
        
    with col_logo:
        logo = "Kiloutou_logo.jpg"
        st.image(logo, width=73)

    
    st.header("1. Analyse client")
    
    
    # tableau
    
    # Collecte des données
    df_table = pd.read_csv(url, delimiter=";").reset_index(drop=True)
    
    # Créer des colonnes pour les listes déroulantes
    col_space, col_country, col_space, col_category, col_space, col_client, col_space = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    
    # Liste déroulante pour le pays
    with col_country:
        selected_country = st.selectbox('Sélectionnez le pays', df_table['Pays/Région'].unique(), index=None, placeholder="Choisir un pays",)
    
    # Liste déroulante pour la catégorie
    with col_category:
        selected_category = st.selectbox('Sélectionnez la catégorie', df_table['Catégorie'].unique(), index=None, placeholder="Choisir une catégorie",)
    
    # Liste déroulante pour le client
    with col_client:
        selected_client = st.selectbox('Sélectionnez le client', df_table['Nom du client'].unique(), index=None, placeholder="Choisir un client",)
    
    # Sélectionner les colonnes à afficher dans le DataFrame
    selected_columns_table = ['Catégorie', 'Date de commande', 'ID client', 'Nom du client', 'Nom du produit', 'Pays/Région', 'Segment', 'Statut des expéditions', 'Ville', 'Quantité', 'Remise', 'Ventes']
    
    # Appliquer les filtres
    df_filtre = df_table[(df_table['Pays/Région'] == selected_country) & (df_table['Catégorie'] == selected_category) & (df_table['Nom du client'] == selected_client)]
    
    df_filtre.reset_index(drop=True, inplace=True)
    
    # Définir une variable pour vérifier si les listes déroulantes ont été sélectionnées
    selection_effectuee = False
    
    # Condition pour vérifier si les éléments nécessaires sont sélectionnés
    if selected_country is not None and selected_category is not None and selected_client is not None:
        selection_effectuee = True
    
    # Condition pour afficher le tableau uniquement si la sélection a été effectuée
    if selection_effectuee:
        # Afficher un graphique (vous pouvez ajuster le style selon vos préférences)
        fig = go.Figure(data=[go.Table(
            columnorder=list(range(len(selected_columns_table))),
            columnwidth=[120, 150, 120, 120, 150, 120, 120, 180, 180, 120, 120, 120],
            header=dict(
                values=selected_columns_table,
                font=dict(size=14, color='white'),
                fill_color='#fcc200',
                line_color='#000000',
                align=['center'],
                height=30
            ),
            cells=dict(
                values=[df_filtre[K].tolist() for K in selected_columns_table],
                font=dict(size=14),
                align=['center'],
                line_color='#000000',
                fill_color='#f3f2f2',
                height=30))
        ])
        
        fig.update_layout(height=400, margin=dict(t=0, b=30))
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    
    col_gauge1, col_gauge2, col_gauge3 = st.columns([1,1,1])
    
    if selection_effectuee:
        with col_gauge1:
            df_filtre['Remise'] = df_filtre['Remise'].str.replace('[^\d.]', '', regex=True).astype(float)
    
            # Calcul de la somme des remises accordées à un client
            somme_remises_client = df_filtre['Remise'].sum()
    
            # Formater la valeur de la jauge pour inclure le symbole de pourcentage
            valeur_jauge_formatee = f"{somme_remises_client:.2f}%"
    
            couleur_jauge = "red" 
            
            if somme_remises_client > 25 :
                couleur_jauge = "green"
                
    
            # Création d'une jauge dynamique avec Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=somme_remises_client,
                number={'suffix': '%'},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cumul des remises accordées"},
                gauge={'axis': {'range': [0, 200]},
                       'steps': [
                           {'range': [0, 50], 'color': "#faf1b7"},
                           {'range': [50, 100], 'color': "#f7e888"},
                           {'range': [100, 150], 'color': "#ffd54d"},
                           {'range': [150, 200], 'color': "#fcc200"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': somme_remises_client}
                       }
            ))
    
            fig_gauge.update_traces(gauge=dict(bar=dict(color=couleur_jauge)))
    
            fig_gauge.update_layout(
                height=200,
                font=dict(size=16),
                margin=dict(l=10, r=10, t=60, b=10, pad=8),
            )
    
            # Affichage de la jauge sous le tableau existant
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    
    
            
    
            with col_gauge2:
                somme_quantites_client = df_filtre['Quantité'].sum()
    
                couleur_jauge = "red" 
            
                if somme_quantites_client > 20 :
                    couleur_jauge = "green"
    
                
                # Création d'une jauge dynamique avec Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=somme_quantites_client,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Nombre d'articles vendus"},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 25], 'color': "#faf1b7"},
                               {'range': [25, 50], 'color': "#f7e888"},
                               {'range': [50, 75], 'color': "#ffd54d"},
                               {'range': [75, 100], 'color': "#fcc200"}],
                           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': somme_quantites_client}
                           }
                ))
    
        
                fig_gauge.update_traces(gauge=dict(bar=dict(color=couleur_jauge)))
        
                fig_gauge.update_layout(
                    height=200,
                    font=dict(size=16),
                    margin=dict(l=10, r=10, t=60, b=10, pad=8),
                )
                
                # Affichage de la jauge sous le tableau existant
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    
    
    
            
    
            with col_gauge3:
                df_filtre['Ventes'] = df_filtre['Ventes'].astype(str)
    
                # Appliquer la modification sur la colonne 'Ventes'
                df_filtre['Ventes'] = df_filtre['Ventes'].str.replace('[^\d]', '', regex=True)
            
                # Convertir la colonne 'Ventes' en type numérique
                df_filtre['Ventes'] = pd.to_numeric(df_filtre['Ventes'], errors='coerce', downcast='integer')
                
                somme_ventes_client = df_filtre['Ventes'].sum()
    
                couleur_jauge = "red" 
            
                if somme_ventes_client > 2000 :
                    couleur_jauge = "green"
                
    
                # Création d'une jauge dynamique avec Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=somme_ventes_client,
                    number={'suffix': '€'},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Montant global des ventes"},
                    gauge={'axis': {'range': [0, 8000]},
                           'steps': [
                               {'range': [0, 2000], 'color': "#faf1b7"},
                               {'range': [2000, 4000], 'color': "#f7e888"},
                               {'range': [4000, 6000], 'color': "#ffd54d"},
                               {'range': [6000, 8000], 'color': "#fcc200"}],
                           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': somme_ventes_client}
                           }
                ))
    
                fig_gauge.update_traces(gauge=dict(bar=dict(color=couleur_jauge)))
                
                fig_gauge.update_layout(
                    height=200,
                    font=dict(size=16),
                    margin=dict(l=10, r=10, t=60, b=10, pad=8),
                )
                
                # Affichage de la jauge sous le tableau existant
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    
    
    
    
    
    
    
    
    
    
    #2) analyse temporelle
    
    
    st.header("2. Analyses temporelles")
    
    #création de colonnes et attribution de dimensions
    col_dd1, col_sp1, cold_dd2, col_sp2, col_mlt = st.columns([0.5,0.5,0.5,0.5,2])
    
    with col_dd1:
        selected_year = st.selectbox("Sélectionnez N", sorted_years, index=3, placeholder="Choisir N")
    
    with cold_dd2:
        if selected_year in sorted_years_2:
            sorted_years_2.remove(selected_year)
            selected_comparison_year = st.selectbox("Sélectionnez N-*", [year for year in sorted_years_2 if year < selected_year])
    
    with col_mlt:
        available_months = sorted(df['Mois'].unique())
        selected_months = st.multiselect("", available_months, default=available_months)
        filtered_df = df[df['Mois'].isin(selected_months)]
    
    #création de colonnes identiques
    col_sp1, col_clients, col_sp2, col_orders, col_sp3, col_ca, col_sp4= st.columns([0.5, 1.25, 0.5, 1.25, 0.5, 1.25, 0.5])
    
    #calculs
    num_clients = df[df['Année'] == selected_year].drop_duplicates('ID client')['ID client'].count()
    num_orders = len(df[df['Année'] == selected_year]['ID commande'])
    ca_by_year = df[df['Année'] == selected_year]['Ventes'].sum()
    
    #calculs des différences pour comparatif entre N et N-*
    diff_clients = num_clients - df[df['Année'] == selected_comparison_year].drop_duplicates('ID client')['ID client'].count()
    diff_orders = num_orders - len(df[df['Année'] == selected_comparison_year]['ID commande'])
    diff_ca = ca_by_year - df[df['Année'] == selected_comparison_year]['Ventes'].sum()
    
    #conversion des données pour conserver uniquement la partie entière
    diff_clients = int(diff_clients)
    diff_orders = int(diff_orders)
    diff_ca = int(diff_ca)
    
    #affiche le nombre de clients selon l'année
    col_clients.metric(label="Nombre de clients", value=num_clients, delta=diff_clients)
    
    #affiche le nombre de commandes selon l'année + comparatif avec N-*
    col_orders.metric(label="Nombre de commandes", value=num_orders, delta=diff_orders)
    
    #affiche le chiffre d'affaires selon l'année + comparatif avec N-*
    col_ca.metric(label=f"Chiffre d'affaires", value=f"{int(ca_by_year)} €", delta=f"{int(diff_ca)} €")
    
    style_metric_cards()
    
    #graphique qui permet d'observer l'évolution du nombre de clients selon N et N-*
    
    col_v1, col_space, col_v2 = st.columns([2,0.5,2])
    
    with col_v1:
        # Agréger le nombre de clients par mois pour l'année sélectionnée
        monthly_clients_selected_year = filtered_df[filtered_df['Année'] == selected_year].drop_duplicates('ID client').groupby(
        'Mois')['ID client'].count().reset_index()
    
        # Tri des mois dans l'ordre
        sorted_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        monthly_clients_selected_year['Mois'] = pd.Categorical(monthly_clients_selected_year['Mois'], categories=sorted_months, ordered=True)
        monthly_clients_selected_year = monthly_clients_selected_year.sort_values('Mois')
    
        # Agréger le nombre de clients par mois pour l'année de comparaison
        monthly_clients_comparison_year = filtered_df[filtered_df['Année'] == selected_comparison_year].drop_duplicates(
        'ID client').groupby('Mois')['ID client'].count().reset_index()
    
        # Tri des mois dans l'ordre
        monthly_clients_comparison_year['Mois'] = pd.Categorical(monthly_clients_comparison_year['Mois'], categories=sorted_months, ordered=True)
        monthly_clients_comparison_year = monthly_clients_comparison_year.sort_values('Mois')
    
        # Affiche l'évolution du nombre de clients pour N
        fig_clients_evolution = go.Figure()
        fig_clients_evolution.add_trace(go.Scatter(
            x=monthly_clients_selected_year['Mois'],
            y=monthly_clients_selected_year['ID client'],
            mode='lines+markers',
            name=f"{selected_year}",
            line=dict(color='#fcc200'),
            marker=dict(symbol='square', size=8, color='#fcc200')
        ))
         
        # Affiche l'évolution du nombre de clients pour N-*
        fig_clients_evolution.add_trace(go.Scatter(
            x=monthly_clients_comparison_year['Mois'],
            y=monthly_clients_comparison_year['ID client'],
            mode='lines+markers',
            name=f"{selected_comparison_year}",
            line=dict(color='#9b870c'),
            marker=dict(symbol='square', size=8, color='#9b870c')
        ))
    
        target_value = 80
        fig_clients_evolution.add_shape(
            go.layout.Shape(
                type="line",
                x0=monthly_clients_selected_year['Mois'].min(),
                x1=monthly_clients_selected_year['Mois'].max(),
                y0=target_value,
                y1=target_value,
                line=dict(color="black", width=2, dash="dash"),
            )
        )
        
        # Mise en forme
        fig_clients_evolution.update_layout(title=f"Évolution du nombre de clients en {selected_year} et {selected_comparison_year}",
                                           xaxis=dict(title='Mois', tickfont=dict(size=12), title_font=dict(size=12)),
                                           yaxis=dict(title='Nombre de clients', tickfont=dict(size=12), title_font=dict(size=12)),
                                           title_font=dict(size=15),
                                           title_x = 0.2,
                                           height=500,
                                           width=500)
        
        # Affichage
        st.plotly_chart(fig_clients_evolution, use_container_width=True)
    
    
    #graphique qui permet d'observer l'évolution du nombre de clients selon N et N-*
    
    fig_orders_evolution = go.Figure()
    
    # Graphique qui permet d'observer l'évolution du nombre de commandes selon N et N-*
    
    with col_v2:
        # Agréger le nombre de commandes par mois pour l'année sélectionnée
        monthly_orders_selected_year = filtered_df[filtered_df['Année'] == selected_year].groupby('Mois')['ID commande'].count().reset_index()
    
        # Tri des mois dans l'ordre croissant
        monthly_orders_selected_year['Mois'] = pd.Categorical(monthly_orders_selected_year['Mois'], categories=sorted_months, ordered=True)
        monthly_orders_selected_year = monthly_orders_selected_year.sort_values('Mois')
    
        # Agréger le nombre de commandes par mois pour l'année de comparaison
        monthly_orders_comparison_year = filtered_df[filtered_df['Année'] == selected_comparison_year].groupby('Mois')['ID commande'].count().reset_index()
    
        # Tri des mois dans l'ordre croissant
        monthly_orders_comparison_year['Mois'] = pd.Categorical(monthly_orders_comparison_year['Mois'], categories=sorted_months, ordered=True)
        monthly_orders_comparison_year = monthly_orders_comparison_year.sort_values('Mois')
        
        # Ajustez la taille des barres ici
        bar_width = 0.3
    
        # Affiche l'évolution du nombre de commandes pour N-*
        fig_orders_evolution.add_trace(go.Bar(
            x=monthly_orders_comparison_year['Mois'],
            y=monthly_orders_comparison_year['ID commande'],
            name=f"{selected_comparison_year}",
            text=monthly_orders_comparison_year['ID commande'],
            textposition='outside',
            marker=dict(color='#f7e98e', line=dict(width=2, color='black')),
            width=bar_width,
        ))
    
        # Affiche l'évolution du nombre de commandes pour N
        fig_orders_evolution.add_trace(go.Bar(
            x=monthly_orders_selected_year['Mois'],
            y=monthly_orders_selected_year['ID commande'],
            name=f"{selected_year}",
            text=monthly_orders_selected_year['ID commande'],
            textposition='outside',
            marker=dict(color='#fcc200', line=dict(width=2, color='black')),
            width=bar_width,
        ))
    
        target_value = 150  # Remplacez cela par la valeur cible souhaitée
        fig_orders_evolution.add_shape(
            go.layout.Shape(
                type="line",
                x0=monthly_orders_comparison_year['Mois'].min(),
                x1=monthly_orders_comparison_year['Mois'].max(),
                y0=target_value,
                y1=target_value,
                line=dict(color="black", width=2, dash="dash"),
            )
        )
    
        # Mise à jour de la mise en forme
        fig_orders_evolution.update_layout(barmode='group', title=f"Évolution du nombre de commandes en {selected_year} et {selected_comparison_year}",
                                           xaxis=dict(title='Nombre de commandes', tickfont=dict(size=12), title_font=dict(size=12)),
                                           yaxis=dict(title='Mois', tickfont=dict(size=12), title_font=dict(size=12)),
                                           title_font=dict(size=15),
                                           title_x=0.2,
                                           height=500,
                                           width=500)
    
        # Affichage
        st.plotly_chart(fig_orders_evolution, use_container_width=True)
    
       
    
    
    
    
    
    
    
    
    
    st.header("3. Analyses géographiques")
    
    
    col_country, col_space = st.columns([0.5, 1])
    
    # Liste déroulante pour le pays
    with col_country:
        selected_pays = st.selectbox('Sélectionnez le pays', df['Pays/Région'].unique(), index=None, placeholder=" ")
    
    selection = False
    
    if selected_pays is not None:
        selection = True
    
    data_f = df[df['Pays/Région'] == selected_pays]
    
    # Colonne pour le classement par pays des 5 produits les plus achetés
    col_class, col_pie = st.columns([1, 1])
    
    with col_class:
        if selection:
            # Grouper par produit et calculer la quantité totale achetée
            top_products = data_f.groupby('Nom du produit')['Quantité'].sum().reset_index()
    
            # Trier par quantité croissante et sélectionner les 5 premiers produits
            top_products = top_products.sort_values(by='Quantité', ascending=True).tail(5)
    
            target_value = top_products['Quantité'].mean()
            
            colors = ['#faf1b7', '#f7e888', '#ffdd1a', '#ffd54d', '#fcc200']
    
            fig = go.Figure()
    
            fig.add_trace(go.Bar(
                y=top_products['Nom du produit'],
                x=top_products['Quantité'],
                orientation='h',
                marker=dict(color=colors),
                text=top_products['Quantité'],
                textposition='outside',
            ))
    
            fig.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=target_value,
                    x1=target_value,
                    y0=0,
                    y1=len(top_products),
                    line=dict(color='black', dash='dash', width=2),
                )
            )
    
            fig.update_layout(
                title='Classement des 5 produits les plus achetés',
                yaxis=dict(title='Produit', tickfont=dict(size=12)),
                xaxis=dict(title='Quantité achetée', tickfont=dict(size=12)),
                title_x=0.25,
                title_font=dict(size=15),
                height=300,
                width=300,
                margin=dict(t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    
    with col_pie :
        quantity_by_category = data_f.groupby('Catégorie')['Quantité'].sum().reset_index()
            
        colors = ['#fcc200', '#ffe033', '#f7e98e']
        fig = px.pie(quantity_by_category, values='Quantité', names='Catégorie',
                     color_discrete_sequence=colors)
            
        fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
        
        fig.update_layout(title='Quantités vendues par catégorie',
                          title_x=0.25,
                          title_font=dict(size=15),
                          height=300,
                          width=300,
                          margin=dict(t=40, b=30, l=100)
    
        )
        
        if selection :
            st.plotly_chart(fig, use_container_width=True)
    
    
    
    # agréger le nombre de clients par pays
    clients_by_country = df.drop_duplicates(subset=['ID client', 'Pays/Région']).groupby('Pays/Région')['ID client'].count().reset_index()

    # récupérer le nombre de clients pour le pays sélectionné
    num_clients = clients_by_country[clients_by_country['Pays/Région'] == selected_pays]['ID client'].values[0]
    
    # fusionner les données agrégées avec les données filtrées
    merged_data = pd.merge(data_f, clients_by_country, how='left', on='Pays/Région')
    
    # icône personnalisée pour représenter un client (ici l'exemple c'est Kiloutou)
    icon_path = 'Kiloutou_logo.jpg'
    client_icon = folium.CustomIcon(icon_image=icon_path, icon_size=(20, 20))
    
    if selection:
        # définition d'une localisation initiale
        my_map = folium.Map(location=[merged_data['Latitude'].iloc[0], merged_data['Longitude'].iloc[0]], zoom_start=5.5)
    
        # ajoutez un seul marqueur pour représenter le pays avec le nombre de clients dans l'infobulle
        folium.Marker([merged_data['Latitude'].iloc[0], merged_data['Longitude'].iloc[0]],
                      popup=f"Nombre de clients: {num_clients}",
                      icon=client_icon).add_to(my_map)
    
        st_folium(my_map, width=1410, height=600)
    
    
    
    
    
    
    #new_dfs, code = spreadsheet(url)



if selected3 == "Import":
    @st.cache_data
    def load_data(file):
        dfo = pd.read_csv(file, delimiter=";")
        return dfo

    uploaded_file = st.file_uploader("Choisir un fichier")

    if uploaded_file is None:
        st.info("Veuillez choisir un fichier à importer")
        st.stop()
    
    dfo = load_data(uploaded_file)
    
    st.dataframe(dfo, 
                 width=1426,
                 column_config={
                    "ID ligne" : st.column_config.NumberColumn(format="%d")
                 },
                )

    col_metric, col_space, col_bar = st.columns([1,0.2,1])
    
    with col_metric:
        
        def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
            fig = go.Figure()
        
            fig.add_trace(
                go.Indicator(
                    value=value,
                    gauge={"axis": {"visible": False}},
                    number={
                        "prefix": prefix,
                        "suffix": suffix,
                        "font.size": 28,
                    },
                    title={
                        "text": label,
                        "font": {"size": 24},
                    },
                )
            )
            
            if show_graph:
                fig.add_trace(
                    go.Scatter(
                        y=random.sample(range(0, 101), 50),
                        hoverinfo="skip",
                        fill="tozeroy",
                        fillcolor=color_graph,
                        line={
                            "color": color_graph,
                        },
                    )
                )
        
            fig.update_xaxes(visible=False, fixedrange=True)
            fig.update_yaxes(visible=False, fixedrange=True)
            fig.update_layout(
                margin=dict(t=30, b=0),
                showlegend=False,
                plot_bgcolor="white",
                height=100,
            )
        
            st.plotly_chart(fig, use_container_width=True)

        
        dfo['Ventes'] = dfo['Ventes'].str.replace('[^\d]', '', regex=True)
        dfo['Ventes'] = pd.to_numeric(dfo['Ventes'], errors='coerce', downcast='integer')
        dfo['Prévision des ventes'] = dfo['Prévision des ventes'].str.replace('[^\d]', '', regex=True)
        dfo['Prévision des ventes'] = pd.to_numeric(dfo['Prévision des ventes'], errors='coerce', downcast='integer')


        df_2023 = dfo[pd.to_datetime(dfo['Date de commande'], format='%d/%m/%Y').dt.year == 2023]
        chiffre_affaires_reel_2023 = df_2023['Ventes'].sum()
        chiffre_affaires_previsionnel_2023 = df_2023['Prévision des ventes'].sum()

        st.subheader("")
        
        plot_metric(
            "Chiffre d'affaires réel", 
            chiffre_affaires_reel_2023, 
            suffix="€", 
            show_graph=True, 
            color_graph="rgba(252, 194, 0, 0.6)"
        )

        st.subheader("")
        st.subheader("")
        st.subheader("")

        plot_metric(
            "Chiffre d'affaires prévisionnel", 
            chiffre_affaires_previsionnel_2023, 
            suffix="€", 
            show_graph=True, 
            color_graph="rgba(252, 194, 0, 0.6)"
        )
        

        with col_bar:
            # Trier les données par quantité décroissante
            dfo = dfo.sort_values(by=['Quantité'], ascending=False)

            # Grouper les quantités vendues par pays et trier en ordre décroissant
            data = dfo.groupby('Pays/Région')['Quantité'].sum().reset_index()
            data = data.sort_values(by='Quantité', ascending=False)

            # Créer le graphique en barres avec Plotly
            fig = px.bar(data, x='Pays/Région', y='Quantité', color='Quantité', color_continuous_scale=['#ffe680', '#fcc200'], labels={'Quantité': 'Quantité vendue', 'Pays/Région': 'Pays'})

            fig.update_layout(yaxis_tickformat='%d')
            fig.update_layout(title='Quantités vendues par pays', title_x = 0.3)
            
            # Afficher le graphique avec Streamlit
            st.plotly_chart(fig)






if selected3 == "OpenAI":
    
    #######################################
    # PREREQUISITES
    #######################################


    headers = {
        "authorization" : st.secrets["OPENAI_API_KEY"],
        "authorization" : st.secrets["OPENAI_ASSISTANT_ID"],
        "authorization" : st.secrets["MAPBOX_TOKEN"]
    }
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]
    
    assistant_state = "assistant"
    thread_state = "thread"
    conversation_state = "conversation"
    last_openai_run_state = "last_openai_run"
    map_state = "map"
    markers_state = "markers"
    
    user_msg_input_key = "input_user_msg"

    #######################################
    # SESSION STATE SETUP
    #######################################

    if (assistant_state not in st.session_state) or (thread_state not in st.session_state):
        st.session_state[assistant_state] = client.beta.assistants.retrieve(assistant_id)
        st.session_state[thread_state] = client.beta.threads.create()
    
    if conversation_state not in st.session_state:
        st.session_state[conversation_state] = []
    
    if last_openai_run_state not in st.session_state:
        st.session_state[last_openai_run_state] = None
    
    if map_state not in st.session_state:
        st.session_state[map_state] = {
            "latitude": 50.60272,
            "longitude": 3.13381,
            "zoom": 16,
        }
    
    if markers_state not in st.session_state:
        st.session_state[markers_state] = None

    #######################################
    # TOOLS SETUP
    #######################################


    def update_map_state(latitude, longitude, zoom):
        """OpenAI tool to update map in-app
        """
        st.session_state[map_state] = {
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
        }
        return "Map updated"
    
    
    def add_markers_state(latitudes, longitudes, labels):
        """OpenAI tool to update markers in-app
        """
        st.session_state[markers_state] = {
            "lat": latitudes,
            "lon": longitudes,
            "text": labels,
        }
        return "Markers added"
    
    
    tool_to_function = {
        "update_map": update_map_state,
        "add_markers": add_markers_state,
    }

    #######################################
    # HELPERS
    #######################################


    def get_assistant_id():
        return st.session_state[assistant_state].id
    
    
    def get_thread_id():
        return st.session_state[thread_state].id
    
    
    def get_run_id():
        return st.session_state[last_openai_run_state].id
    
    
    def on_text_input(status_placeholder):
        """Callback method for any chat_input value change
        """
        if st.session_state[user_msg_input_key] == "":
            return
    
        client.beta.threads.messages.create(
            thread_id=get_thread_id(),
            role="user",
            content=st.session_state[user_msg_input_key],
        )
        st.session_state[last_openai_run_state] = client.beta.threads.runs.create(
            assistant_id=get_assistant_id(),
            thread_id=get_thread_id(),
        )
    
        completed = False
    
        # Polling
        with status_placeholder.status("Computing Assistant answer") as status_container:
            st.write(f"Launching run {get_run_id()}")
    
            while not completed:
                run = client.beta.threads.runs.retrieve(
                    thread_id=get_thread_id(),
                    run_id=get_run_id(),
                )
    
                if run.status == "requires_action":
                    tools_output = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        f = tool_call.function
                        print(f)
                        f_name = f.name
                        f_args = json.loads(f.arguments)
    
                        st.write(f"Launching function {f_name} with args {f_args}")
                        tool_result = tool_to_function[f_name](**f_args)
                        tools_output.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": tool_result,
                            }
                        )
                    st.write(f"Will submit {tools_output}")
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=get_thread_id(),
                        run_id=get_run_id(),
                        tool_outputs=tools_output,
                    )
    
                if run.status == "completed":
                    st.write(f"Completed run {get_run_id()}")
                    status_container.update(label="Assistant is done", state="complete")
                    completed = True
    
                else:
                    time.sleep(0.1)
    
        st.session_state[conversation_state] = [
            (m.role, m.content[0].text.value)
            for m in client.beta.threads.messages.list(get_thread_id()).data
        ]
    
    
    def on_reset_thread():
        client.beta.threads.delete(get_thread_id())
        st.session_state[thread_state] = client.beta.threads.create()
        st.session_state[conversation_state] = []
        st.session_state[last_openai_run_state] = None
    
    #######################################
    # MAIN
    #######################################

    st.title('Assistant OpenAI')
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        with st.container():
            for role, message in st.session_state[conversation_state]:
                with st.chat_message(role):
                    st.write(message)
        status_placeholder = st.empty()
    
    with right_col:
        fig = go.Figure(
            go.Scattermapbox(
                mode="markers",
            )
        )
        if st.session_state[markers_state] is not None:
            fig.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=24,
                        color="red",
                    ),
                    lat=st.session_state[markers_state]["lat"],
                    lon=st.session_state[markers_state]["lon"],
                    text=st.session_state[markers_state]["text"],
                )
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox=dict(
                accesstoken=st.secrets["MAPBOX_TOKEN"],
                center=go.layout.mapbox.Center(
                    lat=st.session_state[map_state]["latitude"],
                    lon=st.session_state[map_state]["longitude"],
                ),
                pitch=0,
                zoom=st.session_state[map_state]["zoom"],
            ),
            height=600,
        )
        st.plotly_chart(
            fig, config={"displayModeBar": False}, use_container_width=True, key="plotly"
        )
    
    st.chat_input(
        placeholder="Posez votre question ici",
        key=user_msg_input_key,
        on_submit=on_text_input,
        args=(status_placeholder,),
    )





def load_data():
    if os.path.exists("tasks_data.csv"):
        return pd.read_csv("tasks_data.csv")
    else:
        return pd.DataFrame([
            {"Tâches": "Chargement des données sur Snowflake", "Personnes Assignées": 2, "Durée": "4h", "Statut": "en cours",
             "Durée restante": "2h"},
        ])

def save_data(data):
    data.to_csv("tasks_data.csv", index=False)

if "tasks_df" not in st.session_state:
    st.session_state.tasks_df = load_data()

if selected3 == "Tâches":
    st.title("Gestion des tâches")
    edited_df = st.data_editor(st.session_state.tasks_df, width=1426, height=600, num_rows="dynamic")
    st.session_state.tasks_df = edited_df
    save_data(edited_df)

    if "Personnes Assignées" in edited_df.columns and "Statut" in edited_df.columns:
        #Initialisation de Personnes Assignées à 0 pour une nouvelle ligne 
        edited_df["Personnes Assignées"].fillna(0, inplace=True)
        edited_df["Personnes Assignées"] = edited_df["Personnes Assignées"].astype(int)
        
        col_1, col_space, col_2, col_3, col_space = st.columns([0.5,0.5,0.5,0.5,0.5])

        tot_effectif = 20
        assigned_persons = edited_df["Personnes Assignées"].sum()
        available_persons = tot_effectif - assigned_persons
    
        with col_1:
            st.metric(label="Effectif total", value=tot_effectif)
            st.metric(label="Personnes assignées à des tâches", value=assigned_persons)
            st.metric(label="Personnes disponibles", value=available_persons)
    
        style_metric_cards()
    
        total_tasks = edited_df.shape[0]
        completed_tasks = (edited_df["Statut"] == "terminée").sum()
        remaining_tasks = total_tasks - completed_tasks
    
        with col_2:
            st.metric(label="Nombre total de tâches", value=total_tasks)
            st.metric(label="Tâches terminées", value=completed_tasks)
            st.metric(label="Tâches restantes", value=remaining_tasks)
    
        style_metric_cards()




if selected3 == "Tests":
    st.header("1. Analyse client")
    st.subheader("")
    st.subheader("")
    
    # Collecte des données
    df_table = pd.read_csv(url, delimiter=";").reset_index(drop=True)
    
    # Sélectionner les colonnes à afficher dans le DataFrame
    selected_columns_table = ['Catégorie', 'Date de commande', 'ID client', 'Nom du client', 'Nom du produit', 'Pays/Région', 'Segment', 'Statut des expéditions', 'Ville', 'Quantité', 'Remise', 'Ventes']

    # Filtrer le DataFrame avec les colonnes sélectionnées
    df_filtered = df_table[selected_columns_table].copy()  # Assurez-vous de copier le DataFrame pour éviter les modifications accidentelles
    
    # Nettoyer la colonne "Ventes"
    df_filtered['Ventes'] = df_filtered['Ventes'].str.replace('[^\d]', '', regex=True)
    df_filtered['Ventes'] = pd.to_numeric(df_filtered['Ventes'], errors='coerce', downcast='integer')
    
    df_filtered['Ventes'] = df_filtered['Ventes'].astype(str)
    df_filtered['Date de commande'] = pd.to_datetime(df_filtered['Date de commande'], format='%d/%m/%Y')

    
    # Afficher le DataFrame dans Streamlit avec le ProgressColumn pour la colonne "Ventes"
    st.data_editor(
    df_filtered,
    column_config={
        "Ventes": st.column_config.ProgressColumn(
            "Volume des ventes",
            format="%f€",
            min_value=0,
            max_value=8000,
        ),
        "Date de commande": st.column_config.DateColumn(
            "Date de commande",
            format="DD.MM.YYYY",
            step=1,
        ),
    },
    hide_index=True,
    )    
    







if selected3 == "Elements" :
    with elements("boutton"):
        mui.Button(
        "TEXT"
    )

        with mui.Button(variant="text"):
            mui.Button("TEXT2")
     
    with elements("properties"):
        with mui.Paper(elevation=3, variant="outlined", square=True):
            mui.TextField(
                label="My text input",
                defaultValue="Type here",
                variant="outlined",
            )























        
    
    
