"""
Script de test de l'environnement et des dépendances
Exécutez ce script pour vérifier que tout fonctionne correctement
"""

import sys
import importlib
import subprocess
import os

def test_python_environment():
    """Teste l'environnement Python"""
    print("=" * 60)
    print("🧪 TEST DE L'ENVIRONNEMENT PYTHON")
    print("=" * 60)
    
    # Version de Python
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Chemins
    print(f"Python path: {sys.path}")
    
    return True

def test_imports():
    """Teste l'importation de toutes les bibliothèques nécessaires"""
    print("\n" + "=" * 60)
    print("📚 TEST DES IMPORTS")
    print("=" * 60)
    
    packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'xgboost',
        'joblib',
        'matplotlib',
        'seaborn',
        'imblearn'
    ]
    
    all_imports_ok = True
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            print(f"✅ {package:20} version: {getattr(module, '__version__', 'N/A')}")
        except ImportError as e:
            print(f"❌ {package:20} ERREUR: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_data_files():
    """Teste l'accès aux fichiers de données"""
    print("\n" + "=" * 60)
    print("📁 TEST DES FICHIERS DE DONNÉES")
    print("=" * 60)
    
    data_files = [
        'data/churn-bigml-80.csv',
        'data/churn-bigml-20.csv'
    ]
    
    all_files_ok = True
    
    for file_path in data_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path:30} taille: {file_size} octets")
        else:
            print(f"❌ {file_path:30} FICHIER NON TROUVÉ")
            all_files_ok = False
    
    return all_files_ok

def test_model_pipeline():
    """Teste les fonctions du pipeline de modèle"""
    print("\n" + "=" * 60)
    print("🔧 TEST DU PIPELINE DE MODÈLE")
    print("=" * 60)
    
    try:
        from model_pipeline import (
            prepare_data, 
            preprocess_data, 
            train_model, 
            load_model, 
            evaluate_model, 
            save_model
        )
        print("✅ Toutes les fonctions du pipeline sont importables")
        
        # Test de préparation des données
        try:
            X_train, X_test, y_train, y_test = prepare_data('data/churn-bigml-80.csv', 'data/churn-bigml-20.csv')
            if X_train is not None:
                print("✅ Fonction prepare_data() fonctionne")
            else:
                print("❌ prepare_data() a retourné None")
                return False
        except Exception as e:
            print(f"❌ Erreur dans prepare_data(): {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'importation du pipeline: {e}")
        return False

def test_dependencies_versions():
    """Affiche les versions des dépendances installées"""
    print("\n" + "=" * 60)
    print("📦 VERSIONS DES DÉPENDANCES")
    print("=" * 60)
    
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'joblib',
        'matplotlib',
        'seaborn',
        'imbalanced-learn'
    ]
    
    for package in packages:
        try:
            result = subprocess.run([
                sys.executable, '-c', 
                f'import {package} as p; print(f"{package}: {{p.__version__}}")'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"📦 {result.stdout.strip()}")
            else:
                print(f"❌ {package}: Impossible de récupérer la version")
                
        except Exception as e:
            print(f"❌ {package}: Erreur - {e}")

def main():
    """Fonction principale de test"""
    print("🚀 DÉMARRAGE DES TESTS D'ENVIRONNEMENT")
    print("Ce script vérifie que tout est correctement configuré.\n")
    
    # Exécution des tests
    env_ok = test_python_environment()
    imports_ok = test_imports()
    files_ok = test_data_files()
    pipeline_ok = test_model_pipeline()
    
    # Affichage des versions
    test_dependencies_versions()
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    if all([env_ok, imports_ok, files_ok, pipeline_ok]):
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        print("Votre environnement est correctement configuré.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Veuillez corriger les problèmes ci-dessus avant de continuer.")
    
    print("\n💡 Conseils:")
    print("1. Assurez-vous que votre environnement virtuel est activé")
    print("2. Vérifiez que tous les fichiers de données existent")
    print("3. Si des imports échouent, exécutez: pip install -r requirements.txt")
    print("4. Redémarrez VSCode après l'activation du venv")

if __name__ == "__main__":
    main()