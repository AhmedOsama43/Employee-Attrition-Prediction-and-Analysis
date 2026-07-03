from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate
from src.recommendation_engine import generate_recommendation
from src.visualization import (
    plot_attrition_distribution,
    plot_income_vs_attrition,
    plot_overtime,
    plot_job_satisfaction,
    plot_correlation
)

def main():
    # Load Data
    df = load_data("data/employee_attrition.csv")

    # Data Visualization
    print("\n Running Data Visualization...\n")

    plot_attrition_distribution(df)
    plot_income_vs_attrition(df)
    plot_overtime(df)
    plot_job_satisfaction(df)

    # Preprocessing
    df = preprocess_data(df)

    # Correlation
    plot_correlation(df)

    # Training
    model, X_test, y_test = train_model(df)

    # Evaluation
    evaluate(model, X_test, y_test)

    # Sample Prediction (High Risk)
    print("\n" + "="*50)
    print("High-Risk Employee Prediction & Recommendation")
    print("="*50)

    # Get High Risk Employee 
    probs = model.predict_proba(X_test)[:, 1]
    max_index = probs.argmax()

    sample = X_test.iloc[max_index]
    prob = probs[max_index]

    # Convert DataFrame
    sample_df = sample.to_frame().T

    # Threshold
    threshold = 0.4
    prediction = 1 if prob >= threshold else 0

    print(f"\nAttrition Probability: {prob:.2f}")
    print(f"Prediction (0=Stay, 1=Leave): {prediction}")

    # Recommendation System
    recommendations = generate_recommendation(sample, prob)

    print("\nRecommendations:")
    if recommendations:
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("No recommendations needed.")


if __name__ == "__main__":
    main()