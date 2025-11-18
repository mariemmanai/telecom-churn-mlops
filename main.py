from model_pipeline import (
    prepare_data, 
    preprocess_data, 
    train_model, 
    load_model, 
    evaluate_model, 
    save_model, 
    predict_new_data
)
import pandas as pd


def main():
    print("🚀 Démarrage du pipeline de prédiction de churn...")
    
    # 1. Test de réentraînement

    # Chargement des données
    X_train, X_test, y_train, y_test = prepare_data('data/churn-bigml-80.csv', 'data/churn-bigml-20.csv')
    
    # Prétraitement SANS rééchantillonnage pour aller plus vite
    X_train_processed, X_test_processed, y_resampled, scaler, encoder_state, encoder_area = preprocess_data(
            X_train, X_test, y_train)
    
    # Réentraînement du modèle
    model = train_model(X_train_processed, y_resampled)
    
    # Évaluation
    evaluate_model(model, X_test_processed, y_test)
    
    # Sauvegarde du nouveau modèle
    save_model(model, scaler, encoder_state, encoder_area)
    print("✅ Réentraînement terminé avec succès!")
    

    #*****************************************************

    # 2. Test d'évaluation

    # Chargement des données
    X_train, X_test, y_train, y_test = prepare_data('data/churn-bigml-80.csv', 'data/churn-bigml-20.csv')
    
    # Prétraitement
    X_train_processed, X_test_processed, y_resampled, scaler, encoder_state, encoder_area = preprocess_data(
        X_train, X_test, y_train
    )
    
    # Chargement du modèle pré-entraîné
    model = load_model()
    
    # Évaluation
    evaluate_model(model, X_test_processed, y_test)
    print("✅ Évaluation terminée avec succès!")



    # *************************************************
    # 3. Test de prédiction
    
    # Chargement du modèle pré-entraîné
    model = load_model()
    
    # Données de test pour prédiction
    sample_data = pd.DataFrame({
        'State': ['KS'],
        'Account length': [128],
        'Area code': [415],
        'International plan': ['No'],
        'Voice mail plan': ['Yes'],
        'Number vmail messages': [25],
        'Total day minutes': [265.1],
        'Total day calls': [110],
        'Total day charge': [45.07],
        'Total eve minutes': [197.4],
        'Total eve calls': [99],
        'Total eve charge': [16.78],
        'Total night minutes': [244.7],
        'Total night calls': [91],
        'Total night charge': [11.01],
        'Total intl minutes': [10.0],
        'Total intl calls': [3],
        'Total intl charge': [2.70],
        'Customer service calls': [1]
    })
    
    predictions, probabilities = predict_new_data(model, sample_data)
    print(f"📊 Prédictions: {predictions}")
    print(f"📈 Probabilités: {probabilities}")
    print("✅ Prédiction terminée avec succès!")


if __name__ == "__main__":
    main()