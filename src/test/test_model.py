import pickle
import os
import json

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def load_test_data():
    test_data = json.load(open(os.path.join(cur_file_path, "../../data/test_word_tag.json")))
    words = [word for word, _ in test_data]
    gold_tags = [tag for _, tag in test_data]
    return words, gold_tags


def evaluate_accuracy(pred_y, gold_y):
    correct = 0
    for pred, gold in zip(pred_y, gold_y):
        if pred == gold:
            correct += 1
    return correct / len(pred_y)


if __name__ == "__main__":
    # Load test data
    words, gold_tags = load_test_data()
    # Load trained model
    hmm = pickle.load(open(os.path.join(cur_file_path, "../../models/hmm_model.pkl"), "rb"))

    # Naive prediction
    naive_pred_tags = hmm.naive_predict(words)
    # Evaluate accuracy of naive predictions
    naive_acc = evaluate_accuracy(naive_pred_tags, gold_tags)
    print("Naive accuracy:", naive_acc)

    # Viterbi optimized prediction
    viterbi_pred_tags = hmm.predict(words)
    # Evaluate accuracy of optimized predictions
    viterbi_acc = evaluate_accuracy(viterbi_pred_tags, gold_tags)
    print("Viterbi accuracy:", viterbi_acc)
