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


with col_title:
    st.title("Suivi des ventes de la société")
        
        
with col_logo:
    logo = "Kiloutou_logo.jpg"
    st.image(logo, width=73)

with st.sidebar:
    selected3 = option_menu("Menu", ["Accueil", "Import",  "Tâches", 'Paramètres'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "rgba(243,189,29,0.8)", "border": "1px solid #CCC", "border-left": "0.5rem solid #000000", "border-radius": "5px", "box-shadow": "0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15)", "text-align": "center"},
        "icon": {"color": "black", "font-size": "16px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#f7e98e"},
        "nav-link-selected": {"background-color": "#f7e98e"},
        "nav-link.active" : {"color": "black"}
    }
)
    



#PARTIE Vis'

#1) analyse client


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

        target_value = data_f['Quantité'].mean()
        
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

# fusionner les données agrégées avec les données filtrées
merged_data = pd.merge(data_f, clients_by_country, how='left', on='Pays/Région')

# icône personnalisée pour représenter un client (ici l'exemple c'est Kiloutou)
icon_path = 'Kiloutou_logo.jpg'
client_icon = folium.CustomIcon(icon_image=icon_path, icon_size=(20, 20))

if selection:
    # définition d'une localisation initiale
    my_map = folium.Map(location=[merged_data['Latitude'].iloc[0], merged_data['Longitude'].iloc[0]], zoom_start=4.7)

    # ajoutez un seul marqueur pour représenter le pays avec le nombre de clients dans l'infobulle
    folium.Marker([merged_data['Latitude'].iloc[0], merged_data['Longitude'].iloc[0]],
                  popup=f"Nombre de clients: {num_clients}",
                  icon=client_icon).add_to(my_map)

    st_folium(my_map, width=1410, height=600)






#new_dfs, code = spreadsheet(url)
