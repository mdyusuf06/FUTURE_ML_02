import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_models(df):

    print("\nTraining using correct dataset columns...")


    category_col = "Ticket Type"
    priority_col = "Ticket Priority"

    print("Category column:", category_col)
    print("Priority column:", priority_col)


    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])


    y_category = df[category_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )

    category_model = LogisticRegression(max_iter=300)
    category_model.fit(X_train, y_train)

    pred = category_model.predict(X_test)

    print("\n===== CATEGORY MODEL =====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    y_priority = df[priority_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_priority, test_size=0.2, random_state=42
    )

    priority_model = LogisticRegression(max_iter=300)
    priority_model.fit(X_train, y_train)

    pred2 = priority_model.predict(X_test)

    print("\n===== PRIORITY MODEL =====")
    print("Accuracy:", accuracy_score(y_test, pred2))
    print(classification_report(y_test, pred2))


    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    pickle.dump(category_model, open("category_model.pkl", "wb"))
    pickle.dump(priority_model, open("priority_model.pkl", "wb"))

    print("\n🔥 Models saved successfully!")
