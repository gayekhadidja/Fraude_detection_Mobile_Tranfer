

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import confusion_matrix
import plotly.express as px 
import numpy as np

import plotly.graph_objects as go
import xgboost as xgb
import numpy as np
import joblib
from sklearn.metrics import r2_score

#from pandas_profiling import profile_report
#from ydata_profiling import ProfileReport

#from streamlit_pandas_profiling import st_profile_report
#from sklearn.model_selection import train_test_splitfrom sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#from sklearn.metrics import precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")


# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
    
    
df = pd.read_csv('data_part_1.csv')



#st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
      
    st.write("### Contexte du projet")
    
    st.write("Ce projet s'inscrit dans un contexte de controle des transactions. L'objectif est de prédire si une transaction est fraudulause ou pas à partir de ses caractéristique.")   
    
    st.write("Ce jeu de données contient des transactions de mobile money générées avec le simulateur PaySim. La simulation était basée sur un échantillon de transactions réelles recueillies par une entreprise qui est le fournisseur du service financier mobile actuellement opérationnel dans plus de 14 pays à travers le monde. Les données sont un ensemble de journaux financiers d'un mois d'un service d'argent mobile mis en œuvre dans un pays africain.")
    
    st.write("Le jeu de données contient (suivant l'exemple ci-dessus) : step - correspond à une unité de temps dans le monde réel. Dans ce cas, 1 étape représente 1 heure de temps. Nombre total d'étapes : 744 (simulation sur 30 jours).")
    
    if st.checkbox("Explication des variables") :
        
        st.write("type - CASH-IN, CASH-OUT, DEBIT, PAYMENT et TRANSFER.")
        st.write("amount - montant de la transaction en monnaie locale.")
        st.write ("nameOrig - client ayant initié la transaction.")
        st.write("oldbalanceOrg - solde initial avant la transaction.")
        st.write("newbalanceOrig - nouveau solde après la transaction.")
        st.write("nameDest - client destinataire de la transaction.")
        st.write("oldbalanceDest - solde initial du destinataire avant la transaction. Notez qu'il n'y a pas d'information pour les clients dont le nom commence par M (commerçants).")
        st.write("newbalanceDest - nouveau solde du destinataire après la transaction. Notez qu'il n'y a pas d'information pour les clients dont le nom commence par M (commerçants).")
        st.write("isFraud - Il s'agit des transactions effectuées par des agents frauduleux dans la simulation. Dans ce jeu de données spécifique, le comportement frauduleux des agents vise à tirer profit en prenant le contrôle des comptes clients et en essayant de vider les fonds en les transférant vers un autre compte, puis en les retirant du système.")
        st.write("isFlaggedFraud - Le modèle commercial vise à contrôler les transferts massifs d'un compte à un autre et signale les tentatives illégales. Une tentative illégale dans ce jeu de données est une tentative de transfert de plus de 200 000 dans une seule transaction.")
    #st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si une transaction est fraudulause ou pas.")
    
    #st.image("fraude_detection.jpg")


elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    #st.write(df.info)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
        
    if st.checkbox("Afficher l'ensemble des variable"):
        st.write(df.columns)
    
        
    if st.checkbox("Afficher les différents type de transaction "):
       st.write((df["type"]).unique())
       st.write(df['type'].value_counts())
       
       
    if st.checkbox("Afficher les distributions des type de transaction"):
        # The classes are heavily skewed we need to solve this issue later.
        st.write('CASH_OUT', round(df['type'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
        st.write('PAYMENT', round(df['type'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
        st.write('CASH_IN', round(df['type'].value_counts()[2]/len(df) * 100,2), '% of the dataset')
        st.write('TRANSFER', round(df['type'].value_counts()[3]/len(df) * 100,2), '% of the dataset')
        st.write('DEBIT', round(df['type'].value_counts()[4]/len(df) * 100,2), '% of the dataset')

    if st.checkbox("Distribution des transactions frauduleuse et non frauduleuse  ") :
        
        st.write('Non Frauduleuse', round(df['isFraud'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
        st.write('Frauduleuse', round(df['isFraud'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
    
        
 
elif page == pages[2]:
    st.write("### Analyse de données")

   # Récupération des données Power BI
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiMzRmYWM3MjEtZTQ5ZC00MGY3LTljNmYtYjBlYWFhMjMxZjg0IiwidCI6IjMzNDQwZmM2LWI3YzctNDEyYy1iYjczLTBlNzBiMDE5OGQ1YSIsImMiOjh9"

    # Affichage de l'iframe Power BI dans Streamlit
    st.components.v1.iframe(src=power_bi_url, width=700, height=600, scrolling=True)
    

     
elif page == pages[3]:
    
    data_t = df.copy()
    
    data_cash_trans = data_t.query('type == "TRANSFER" or type == "CASH_OUT"')
    
    var_catégorielle = ['nameOrig', 'nameDest', 'isFlaggedFraud']

    colonnes_existantes = df.columns.tolist()
    colonnes_a_supprimer = [colonne for colonne in var_catégorielle if colonne in colonnes_existantes]

    data_cash_trans = df.drop(columns=colonnes_a_supprimer)
    
    for label, content in data_cash_trans.items():
        if not pd.api.types.is_numeric_dtype(content):
            data_cash_trans[label] = pd.Categorical(content).codes+1
    
    X = data_cash_trans.drop('isFraud', axis=1)
    y = data_cash_trans['isFraud']
    
    
    np.random.seed(37)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    import joblib
    
    XGBClassifier = joblib.load('xgclassifier_model.joblib')

    
    import pickle
    import json
    import xgboost as xgb
    
    class XGBModelWrapper:
        def __init__(self, model):
            self.model = model
    
        def predict(self, data):
            # Adapter la logique de prédiction en fonction de votre modèle XGBoost
            return self.model.predict(data)
    
        def save_model(self, file_name):
            with open(file_name, 'wb') as file:
                pickle.dump(self.model, file)
    
        @classmethod
        def load_model(cls, file_name):
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
            return cls(model)
    
    # Charger le modèle XGBoost depuis le fichier Pickle
    #pickle_in = open('XGBClassier.pickle', 'rb')
    #xgb_model = pickle.load(pickle_in)
    
    import xgboost as xgb
    

    # Assurez-vous que X_test est correctement défini avant cette ligne
    #y_pred = classifier.predict(X_test)
    #redict_train = model.predict(X_train)
    #c_train = confusion_matrix(y_train, predict_train)
    #predict_test = model.predict(X_test)
    #c_test = confusion_matrix(y_test, predict_test)
    
     
    # Charger le modèle
    loaded_xgb_model = xgb.Booster(model_file='XGBClassierTEST.model')

    #classifier = pickle.load(pickle_in) 
    import streamlit as st
    import numpy as np
    
  
    
    def make_prediction(step, type_transaction, amount, newbalanceOrg, oldbalanceDest, isflaggegfraude):
        # Convertir les entrées en nombres (assurez-vous que toutes les valeurs sont numériques)
        inputs = np.array([step, type_transaction, amount, newbalanceOrg, oldbalanceDest, isflaggegfraude]).astype(float)
        
        # Suppose inputs is your input data (DataFrame or list)
        # Convert it to a NumPy array
        inputs_array = np.array(inputs)
        


        # Effectuer la prédiction
        prediction =  XGBClassifier.predict([inputs])
    
        return prediction
    
        import streamlit as st

    def main():
        with st.form("form"):
            html_temp = """
            <div style ="background-color:yellow;padding:13px">
            <h1 style ="color:black;text-align:center;">Fraud detection in mobile money transfer </h1>
            </div>
            """
            st.markdown(html_temp, unsafe_allow_html=True)
    
            # Entrée pour l'heure de la transaction
            step = st.text_input("Heure de la transaction", "")
    
            # Sélection du type de transaction avec une contrainte radio
            type_transaction = st.radio("Sélectionner le type de transaction:", ("Cash_out", "Transfer", "Other"))
    
            # Conversion du type de transaction en entier (1 pour "Cash_out" et 2 pour les autres)
            if type_transaction == "Cash_out":
                type_transaction = 1
            else:
                type_transaction = 2
    
            # Contrôle de saisie pour le montant
            amount = st.text_input("Montant", "")
    
            # Contrôle de saisie pour les soldes
            newbalanceOrg = st.text_input("Solde avant la transaction", "")
            oldbalanceDest = st.text_input("Solde après la transaction du Destinataire", "")
    
            # Détermination de la fraude en fonction du type de transaction et du montant
            if type_transaction == 2 :
                isflaggedfraud = 1
            else:
                isflaggedfraud = 0
    
            # Bouton de soumission du formulaire
            submitted = st.form_submit_button("Prédiction")
    
        if submitted:
            #if st.button("Predict"):
                # Affichage des données saisies
                st.write("Step:", step)
                st.write("Type transaction:", type_transaction)
                st.write("Montant:", amount)
                st.write("Solde avant la transaction:", newbalanceOrg)
                st.write("Solde après la transaction du Destinataire:", oldbalanceDest)
    
                # Effectuer la prédiction
                result = make_prediction(step, type_transaction, amount, newbalanceOrg, oldbalanceDest, isflaggedfraud)
    
                # Afficher le résultat de la prédiction
                if result == 0:
                    st.success('Transaction non frauduleuse')
                elif result == 1:
                    st.error('Transaction frauduleuse')
    
    if __name__ == "__main__":
        main()
    
                        
               # from sklearn.metrics import accuracy_score
                
                # Calcul de l'accuracy (précision)
                #X_test = [[step, type_transaction, amount, newbalanceOrg, oldbalanceOrig, newbalanceDest, oldbalanceDest, isflaggedfraud]]
        
                #if 'X_test' in locals() and 'y_test' in locals():
            # Supposons que X_test soit une liste de listes représentant les entrées de test
                 #   y_pred = [make_prediction(*x) for x in X_test]
                  #  accuracy = accuracy_score(y_test, y_pred)
            #st.write(f"Accuracy: {accuracy}")
            
        import psycopg2
    
    
    def connect_to_db():
        conn = psycopg2.connect(
            host = "localhost",
            port = 5432,
            database = "Fraude_detection",
            user = "postgres",
            password = "localhostpass"
     
        )
        return conn
    
    # Fonction pour insérer les données du formulaire et les résultats de la prédiction dans la base de données
    def insert_data(conn, step, type_transaction, amount, newbalanceOrg, oldbalanceDest, result):
        cursor = conn.cursor()
        print("la connexion est passé")
        cursor.execute("""
            INSERT INTO predictions (step, type_transaction, amount, newbalanceorg, oldbalancedest, result)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (step, type_transaction, amount, newbalanceOrg, oldbalanceDest, result))
        conn.commit()
        cursor.close()
    


