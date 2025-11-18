# tests/test_pipeline.py
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ajoute le chemin du projet pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_pipeline import prepare_data, train_model
import joblib

class TestPipeline(unittest.TestCase):
    
    def test_data_loading(self):
        """Test que les données se chargent correctement"""
        try:
            X_train, X_test, y_train, y_test = prepare_data(
                'data/churn-bigml-80.csv', 
                'data/churn-bigml-20.csv'
            )
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            print("✅ Test de chargement des données réussi")
        except Exception as e:
            self.fail(f"Échec du chargement des données: {e}")
    
    def test_data_shapes(self):
        """Test que les données ont les bonnes dimensions"""
        X_train, X_test, y_train, y_test = prepare_data(
            'data/churn-bigml-80.csv', 
            'data/churn-bigml-20.csv'
        )
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        print("✅ Test des dimensions des données réussi")
    
    def test_model_training(self):
        """Test que le modèle peut s'entraîner avec des données prétraitées"""
        try:
            # Charge les données
            X_train, X_test, y_train, y_test = prepare_data(
                'data/churn-bigml-80.csv', 
                'data/churn-bigml-20.csv'
            )
            
            # Utilise un petit subset pour tester rapidement
            X_train_subset = X_train[:100]
            y_train_subset = y_train[:100]
            
            # Prétraite les données (comme dans ton vrai pipeline)
            X_train_subset_processed = self._preprocess_for_test(X_train_subset)
            
            # Entraîne le modèle
            model = train_model(X_train_subset_processed, y_train_subset)
            self.assertIsNotNone(model)
            print("✅ Test d'entraînement du modèle réussi")
            
        except Exception as e:
            self.fail(f"Échec de l'entraînement du modèle: {e}")
    
    def test_file_existence(self):
        """Test que les fichiers de données existent"""
        self.assertTrue(os.path.exists('data/churn-bigml-80.csv'))
        self.assertTrue(os.path.exists('data/churn-bigml-20.csv'))
        print("✅ Test d'existence des fichiers réussi")
    
    def test_preprocessing(self):
        """Test que le prétraitement fonctionne"""
        try:
            X_train, X_test, y_train, y_test = prepare_data(
                'data/churn-bigml-80.csv', 
                'data/churn-bigml-20.csv'
            )
            
            # Test avec un petit subset
            X_train_subset = X_train[:50]
            X_test_subset = X_test[:50]
            y_train_subset = y_train[:50]
            
            # Vérifie que les données brutes contiennent des colonnes texte
            self.assertTrue('State' in X_train_subset.columns)
            self.assertTrue('International plan' in X_train_subset.columns)
            self.assertEqual(X_train_subset['International plan'].dtype, 'object')  # Doit être du texte
            
            # Prétraite les données
            X_train_processed = self._preprocess_for_test(X_train_subset)
            X_test_processed = self._preprocess_for_test(X_test_subset)
            
            # Vérifie que les colonnes texte originales ont été transformées
            self.assertFalse('State' in X_train_processed.columns)  # Doit être supprimée
            # 'International plan' existe toujours mais est maintenant numérique
            self.assertTrue('International plan' in X_train_processed.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(X_train_processed['International plan']))
            
            # Vérifie que les colonnes one-hot ont été créées
            state_columns = [col for col in X_train_processed.columns if col.startswith('State_')]
            area_columns = [col for col in X_train_processed.columns if col.startswith('Area_code_')]
            self.assertGreater(len(state_columns), 0, "Aucune colonne State créée")
            self.assertGreater(len(area_columns), 0, "Aucune colonne Area code créée")
            
            # Vérifie que les nouvelles colonnes de feature engineering existent
            self.assertTrue('Total calls' in X_train_processed.columns)
            self.assertTrue('Total charge' in X_train_processed.columns)
            self.assertTrue('CScalls Rate' in X_train_processed.columns)
            
            # Vérifie que toutes les colonnes sont numériques
            for col in X_train_processed.columns:
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(X_train_processed[col]),
                    f"La colonne {col} n'est pas numérique après prétraitement. Type: {X_train_processed[col].dtype}"
                )
            
            print("✅ Test de prétraitement réussi")
            
        except Exception as e:
            self.fail(f"Échec du prétraitement: {e}")

    def _preprocess_for_test(self, data):
        """Fonction utilitaire pour prétraiter les données pour les tests"""
        data_processed = data.copy()
        
        # 1. Encodage manuel des variables catégorielles
        data_processed['International plan'] = data_processed['International plan'].map({'No': 0, 'Yes': 1})
        data_processed['Voice mail plan'] = data_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
        
        # 2. Encodage one-hot manuel pour State (simplifié pour les tests)
        state_dummies = pd.get_dummies(data_processed['State'], prefix='State')
        area_dummies = pd.get_dummies(data_processed['Area code'], prefix='Area_code')
        
        # 3. Suppression des colonnes originales et concaténation
        data_processed = data_processed.drop(['State', 'Area code'], axis=1)
        data_processed = pd.concat([data_processed, state_dummies, area_dummies], axis=1)
        
        # 4. Feature engineering (comme dans ton vrai code)
        data_processed['Total calls'] = (
            data_processed['Total day calls'] + 
            data_processed['Total eve calls'] + 
            data_processed['Total night calls'] + 
            data_processed['Total intl calls']
        )
        data_processed['Total charge'] = (
            data_processed['Total day charge'] + 
            data_processed['Total eve charge'] + 
            data_processed['Total night charge'] + 
            data_processed['Total intl charge']
        )
        data_processed['CScalls Rate'] = (
            data_processed['Customer service calls'] / data_processed['Account length']
        )
        data_processed['CScalls Rate'] = data_processed['CScalls Rate'].replace([np.inf, -np.inf], 0)
        
        # 5. Suppression des colonnes corrélées
        correlated_columns = [
            'Total day minutes', 'Total eve minutes', 'Total night minutes', 
            'Total intl minutes', 'Voice mail plan'
        ]
        existing_correlated = [col for col in correlated_columns if col in data_processed.columns]
        data_processed.drop(existing_correlated, axis=1, inplace=True)
        
        return data_processed

if __name__ == '__main__':
    unittest.main()