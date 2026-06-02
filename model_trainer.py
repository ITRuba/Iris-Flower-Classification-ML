from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self):
        print("\n--- Model Training Process ---")
        X = self.df.drop('species', axis=1)
        y = self.df['species']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data split successfully.")
        print(f" - Training samples: {len(self.X_train)}")
        print(f" - Testing samples : {len(self.X_test)}")

    def train_model(self):
        if self.X_train is None:
            print("Error: Please run split_data() first.")
            return

        self.model = LogisticRegression(max_iter=200)
        self.model.fit(self.X_train, self.y_train)
        print("\nModel Selection: ")
        print(" - Model used: Logistic Regression")
        print("\nTraining:")
        print("Model trained successfully.")

    def evaluate_model(self):
        if self.model is None or self.X_test is None:
            print("Error: Model not trained or data not split.")
            return
        print("\n--- Model Evaluation ---")

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f" - Model Accuracy: {accuracy * 100:.2f}%")
