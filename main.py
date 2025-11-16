# main.py
from model_pipeline import (
    load_data, 
    explore_data, 
    prepare_data, 
    train_model, 
    evaluate_model, 
    save_model
)

def main():
    """
    Pipeline principal pour l'entraînement du modèle de churn
    """
    print("🚀 Démarrage du pipeline MLOps - Prédiction de Churn")
    print("=" * 50)
    
    # 1. Chargement des données
    X_train, X_test = load_data(
        'data/churn-bigml-80.csv', 
        'data/churn-bigml-20.csv'
    )
    
    if X_train is None:
        print("❌ Erreur: Impossible de charger les données")
        return
    
    # 2. Exploration des données
    explore_data(X_train, "Données d'entraînement")
    explore_data(X_test, "Données de test")
    
    # 3. Préparation des données
    X_train_processed, X_test_processed, y_train, y_test = prepare_data(X_train, X_test)
    
    # 4. Entraînement du modèle
    model = train_model(X_train_processed, y_train, model_type='random_forest')
    
    # 5. Évaluation du modèle
    accuracy = evaluate_model(model, X_test_processed, y_test)
    
    # 6. Sauvegarde du modèle
    save_model(model, 'models/churn_model.joblib')
    
    print(f"\n🎉 Pipeline terminé avec succès!")
    print(f"📈 Accuracy finale: {accuracy:.4f}")

if __name__ == "__main__":
    main()