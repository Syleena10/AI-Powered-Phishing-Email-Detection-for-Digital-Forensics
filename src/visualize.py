import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test_vec, y_test):
    ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


def plot_top_words(vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    word_importance = list(zip(feature_names, coefficients))

    top_phishing = sorted(word_importance, key=lambda x: x[1], reverse=True)[:10]

    words = [w[0] for w in top_phishing]
    values = [w[1] for w in top_phishing]

    plt.figure()
    plt.barh(words, values)
    plt.title("Top Phishing Words")
    plt.gca().invert_yaxis()
    plt.savefig("outputs/phishing_words.png")
    plt.show()
