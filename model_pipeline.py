import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_data(train_path, test_path):    
    # Chargement
    X_train = pd.read_csv('data/churn-bigml-80.csv')
    X_test = pd.read_csv('data/churn-bigml-20.csv')
    # Conversion de la target en int
    X_train['Churn'] = X_train['Churn'].astype(int)
    X_test['Churn'] = X_test['Churn'].astype(int)
    # Séparation features/target
    y_train = X_train['Churn']
    y_test = X_test['Churn']
    X_train = X_train.drop('Churn', axis=1)
    X_test = X_test.drop('Churn', axis=1)
    return X_train, X_test, y_train, y_test
def load_preprocessors():
        scaler = joblib.load('models/scaler.pkl')
        encoder_state = joblib.load('models/encoder_state.pkl')
        encoder_area = joblib.load('models/encoder_area.pkl')
        return scaler, encoder_state, encoder_area

def preprocess_data(X_train, X_test, y_train):
    scaler, encoder_state, encoder_area = load_preprocessors()
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    print("   - Encodage des variables catégorielles...")
    
    # Encodage manuel des plans
    X_train_processed['International plan'] = X_train_processed['International plan'].map({'No': 0, 'Yes': 1})
    X_test_processed['International plan'] = X_test_processed['International plan'].map({'No': 0, 'Yes': 1})
    X_train_processed['Voice mail plan'] = X_train_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
    X_test_processed['Voice mail plan'] = X_test_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
    
    # Encodage State - Version compatible
    encoded_states_X_train = encoder_state.transform(X_train_processed[['State']])
    encoded_states_X_test = encoder_state.transform(X_test_processed[['State']])
    
    # Création des noms de colonnes manuellement
    state_columns = [f'State_{cat}' for cat in encoder_state.categories_[0]]
    encoded_states_df_X_train = pd.DataFrame(encoded_states_X_train, columns=state_columns)
    encoded_states_df_X_test = pd.DataFrame(encoded_states_X_test, columns=state_columns)
    
    # Encodage Area code - Version compatible
    encoded_area_X_train = encoder_area.transform(X_train_processed[['Area code']])
    encoded_area_X_test = encoder_area.transform(X_test_processed[['Area code']])
    
    # Création des noms de colonnes manuellement
    area_columns = [f'Area code_{cat}' for cat in encoder_area.categories_[0]]
    encoded_area_df_X_train = pd.DataFrame(encoded_area_X_train, columns=area_columns)
    encoded_area_df_X_test = pd.DataFrame(encoded_area_X_test, columns=area_columns)
    
    # Suppression des colonnes originales et concaténation
    X_train_processed = X_train_processed.drop(['State', 'Area code'], axis=1)
    X_test_processed = X_test_processed.drop(['State', 'Area code'], axis=1)
    X_train_processed = pd.concat([X_train_processed, encoded_states_df_X_train, encoded_area_df_X_train], axis=1)
    X_test_processed = pd.concat([X_test_processed, encoded_states_df_X_test, encoded_area_df_X_test], axis=1)
    
    # Le reste de votre code reste inchangé...
    print(" - Feature engineering...")
    X_train_processed['Total calls'] = X_train_processed['Total day calls'] + X_train_processed['Total eve calls'] + X_train_processed['Total night calls'] + X_train_processed['Total intl calls']
    X_test_processed['Total calls'] = X_test_processed['Total day calls'] + X_test_processed['Total eve calls'] + X_test_processed['Total night calls'] + X_test_processed['Total intl calls']
    X_train_processed['Total charge'] = X_train_processed['Total day charge'] + X_train_processed['Total eve charge'] + X_train_processed['Total night charge'] + X_train_processed['Total intl charge']
    X_test_processed['Total charge'] = X_test_processed['Total day charge'] + X_test_processed['Total eve charge'] + X_test_processed['Total night charge'] + X_test_processed['Total intl charge']
    X_train_processed['CScalls Rate'] = X_train_processed['Customer service calls'] / X_train_processed['Account length']
    X_test_processed['CScalls Rate'] = X_test_processed['Customer service calls'] / X_test_processed['Account length']
    
    # Remplacer les infinis par 0
    X_train_processed['CScalls Rate'] = X_train_processed['CScalls Rate'].replace([np.inf, -np.inf], 0)
    X_test_processed['CScalls Rate'] = X_test_processed['CScalls Rate'].replace([np.inf, -np.inf], 0)
    
    # Suppression des colonnes corrélées
    correlated_columns = [
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 
        'Total intl minutes', 'Voice mail plan'
    ]
    existing_correlated = [col for col in correlated_columns if col in X_train_processed.columns]
    X_train_processed.drop(existing_correlated, axis=1, inplace=True)
    X_test_processed.drop(existing_correlated, axis=1, inplace=True)
    
    # Application du scaler
    X_train_scaled = scaler.transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_scaled, X_test_scaled, y_train, scaler, encoder_state, encoder_area

def train_model(X_train, y_train):
    optimal_params = {
        'n_estimators': 50,
        'learning_rate': 0.04213701854582754,
        'max_depth': 10,
        'min_child_weight': 2,
        'subsample': 0.9459370873345296,
        'colsample_bytree': 0.8740236539548397,
        'gamma': 1.798508138696485,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    model = XGBClassifier(**optimal_params)
    model.fit(X_train, y_train)
    return model
def load_model():

        model = joblib.load('models/XGBoost.pkl')        
        # Affiche les hyperparamètres pour vérification
        print("🎯 Hyperparamètres du modèle:")
        print(f"   - n_estimators: {model.n_estimators}")
        print(f"   - learning_rate: {model.learning_rate}")
        print(f"   - max_depth: {model.max_depth}")
        return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n🎯 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred, accuracy

def save_model(model, scaler, encoder_state, encoder_area, path='models/'):    
    # Création du dossier s'il n'existe pas
    os.makedirs(path, exist_ok=True)
    
    # Sauvegarde des objets
    joblib.dump(model, f'{path}/XGBoost.pkl')
    joblib.dump(scaler, f'{path}/scaler.pkl')
    joblib.dump(encoder_state, f'{path}/encoder_state.pkl')
    joblib.dump(encoder_area, f'{path}/encoder_area.pkl')
    
def predict_new_data(model, new_data):
    scaler, encoder_state, encoder_area = load_preprocessors()
    new_data_processed = new_data.copy()
    
    # 1. Encodage des variables catégorielles
    new_data_processed['International plan'] = new_data_processed['International plan'].map({'No': 0, 'Yes': 1})
    new_data_processed['Voice mail plan'] = new_data_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
    
    # 2. Application des encodeurs pré-entraînés
    encoded_states = encoder_state.transform(new_data_processed[['State']])
    encoded_area = encoder_area.transform(new_data_processed[['Area code']])
    
    encoded_states_df = pd.DataFrame(
        encoded_states, 
      columns=[f'State_{cat}' for cat in encoder_state.categories_[0]]
    )
    encoded_area_df = pd.DataFrame(
        encoded_area, 
       columns=[f'Area code_{cat}' for cat in encoder_area.categories_[0]]
    )
    
    new_data_processed = new_data_processed.drop(['State', 'Area code'], axis=1)
    new_data_processed = pd.concat([new_data_processed, encoded_states_df, encoded_area_df], axis=1)
    
    # 3. Feature Engineering
    new_data_processed['Total calls'] = new_data_processed['Total day calls'] + new_data_processed['Total eve calls'] + new_data_processed['Total night calls'] + new_data_processed['Total intl calls']
    new_data_processed['Total charge'] = new_data_processed['Total day charge'] + new_data_processed['Total eve charge'] + new_data_processed['Total night charge'] + new_data_processed['Total intl charge']
    new_data_processed['CScalls Rate'] = new_data_processed['Customer service calls'] / new_data_processed['Account length']
    new_data_processed['CScalls Rate'] = new_data_processed['CScalls Rate'].replace([np.inf, -np.inf], 0)
    
    # 4. Suppression des colonnes corrélées
    correlated_columns = [
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 
        'Total intl minutes', 'Voice mail plan'
    ]
    existing_correlated = [col for col in correlated_columns if col in new_data_processed.columns]
    new_data_processed.drop(existing_correlated, axis=1, inplace=True)
    
    # 5. Application du scaler pré-entraîné
    new_data_scaled = scaler.transform(new_data_processed)
    
    # 6. Prédiction
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    
    print(f"✅ Prédictions terminées - {len(predictions)} échantillons")
    return predictions, probabilities

if __name__ == "__main__":

    model = load_model()
    print("✅ Test de chargement réussi!")
        