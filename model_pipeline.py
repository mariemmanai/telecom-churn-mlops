# model_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(train_path, test_path):
    """
    Charger les données d'entraînement et de test
    
    Args:
        train_path (str): Chemin vers le fichier d'entraînement
        test_path (str): Chemin vers le fichier de test
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = pd.read_csv(train_path)
        X_test = pd.read_csv(test_path)
        print("✅ Données chargées avec succès")
        print(f"📊 Shape train: {X_train.shape}, test: {X_test.shape}")
        return X_train, X_test
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None

def explore_data(df, name="Dataset"):
    """
    Exploration basique des données
    """
    print(f"\n🔍 Exploration de {name}:")
    print(f"Shape: {df.shape}")
    print("\n📈 Informations basiques:")
    print(df.info())
    print("\n📊 Statistiques descriptives:")
    print(df.describe())
    print("\n🎯 Variable cible 'Churn':")
    print(df['Churn'].value_counts())
    
    return df

def prepare_data(X_train, X_test):
    """
    Prétraiter les données : nettoyage, encodage, feature engineering
    """
    print("\n🔄 Début du prétraitement des données...")
    
    # Séparer features et target
    y_train = X_train['Churn']
    y_test = X_test['Churn']
    
    # Supprimer la colonne target des features
    X_train = X_train.drop('Churn', axis=1)
    X_test = X_test.drop('Churn', axis=1)
    
    # 1. Encodage des variables catégorielles
    categorical_cols = ['State', 'International plan', 'Voice mail plan']
    
    # Encoder International plan et Voice mail plan
    X_train['International plan'] = X_train['International plan'].map({'No': 0, 'Yes': 1})
    X_test['International plan'] = X_test['International plan'].map({'No': 0, 'Yes': 1})
    
    X_train['Voice mail plan'] = X_train['Voice mail plan'].map({'No': 0, 'Yes': 1})
    X_test['Voice mail plan'] = X_test['Voice mail plan'].map({'No': 0, 'Yes': 1})
    
    # One-Hot Encoding pour State
    X_train = pd.get_dummies(X_train, columns=['State'], prefix='State')
    X_test = pd.get_dummies(X_test, columns=['State'], prefix='State')
    
    # 2. Feature Engineering (comme dans votre notebook)
    X_train['Total calls'] = X_train['Total day calls'] + X_train['Total eve calls'] + X_train['Total night calls'] + X_train['Total intl calls']
    X_train['Total charge'] = X_train['Total day charge'] + X_train['Total eve charge'] + X_train['Total night charge'] + X_train['Total intl charge']
    X_train['CS calls Rate'] = X_train['Customer service calls'] / X_train['Account length']
    
    X_test['Total calls'] = X_test['Total day calls'] + X_test['Total eve calls'] + X_test['Total night calls'] + X_test['Total intl calls']
    X_test['Total charge'] = X_test['Total day charge'] + X_test['Total eve charge'] + X_test['Total night charge'] + X_test['Total intl charge']
    X_test['CS calls Rate'] = X_test['Customer service calls'] / X_test['Account length']
    
    print("✅ Prétraitement terminé")
    print(f"📊 Nouvelles shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='random_forest'):
    """
    Entraîner un modèle de machine learning
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        model_type: Type de modèle ('random_forest', 'gradient_boosting')
    
    Returns:
        model: Modèle entraîné
    """
    print(f"\n🎯 Entraînement du modèle: {model_type}")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    else:
        raise ValueError("Modèle non supporté")
    
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évaluer les performances du modèle
    """
    print("\n📊 Évaluation du modèle...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    print("\n📋 Rapport de classification:")
    print(classification_report(y_test, y_pred))
    
    print("\n🎯 Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def save_model(model, filepath):
    """
    Sauvegarder le modèle entraîné
    """
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé: {filepath}")

def load_model(filepath):
    """
    Charger un modèle sauvegardé
    """
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé: {filepath}")
    return model

if __name__ == "__main__":
    # Test des fonctions
    print("🧪 Test du module model_pipeline...")