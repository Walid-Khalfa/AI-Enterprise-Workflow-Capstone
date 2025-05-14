from solution_guidance.cslib import fetch_data, train_model

def test_model_accuracy():
    data = fetch_data('cs-train')
    model, accuracy = train_model(data)
    assert accuracy > 0.85