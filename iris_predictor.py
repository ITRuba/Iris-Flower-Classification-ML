import tkinter as tk
from tkinter import messagebox
import pandas as pd


class IrisPredictor:
    def __init__(self, trained_model):
        self.model = trained_model
        self.species_names = ['Setosa', 'Versicolor', 'Virginica']

    def predict_species(self, features):

        prediction = self.model.predict(features)[0]
        return self.species_names[prediction]


class IrisGUI:
    def __init__(self, predictor):
        self.predictor = predictor
        self.window = tk.Tk()
        self.window.title("Iris Flower Predictor")
        self.window.geometry("300x400")

        tk.Label(self.window, text="Iris Species Classifier", font=("Arial", 14, "bold"), pady=10).pack()

        self.labels = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]
        self.entries = []


        for label_text in self.labels:
            tk.Label(self.window, text=label_text).pack(pady=2)
            entry = tk.Entry(self.window)
            entry.pack(pady=5)
            self.entries.append(entry)


        self.predict_btn = tk.Button(self.window, text="Predict Species",
                                     command=self.make_prediction,
                                     bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.predict_btn.pack(pady=20)


        self.result_label = tk.Label(self.window, text="", font=("Arial", 12, "italic"), fg="blue")
        self.result_label.pack(pady=10)

    def make_prediction(self):
        try:
            values = [float(e.get().strip()) for e in self.entries]

            df = pd.DataFrame([values], columns=self.predictor.model.feature_names_in_)

            result = self.predictor.predict_species(df)

            self.result_label.config(
                text=f"Predicted Species:\n{result}"
            )

        except Exception as e:
            messagebox.showerror(
                title="Error",
                message=str(e)
            )
            self.result_label.config(text="")

    def run(self):
        self.window.mainloop()