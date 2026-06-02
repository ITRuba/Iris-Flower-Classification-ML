import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.species_names = ['Setosa', 'Versicolor', 'Virginica']

    def scatter_plot(self):
        colors = ['#2980B9', '#1ABC9C', '#8E44AD']
        for i in range(3):
            subset = self.df[self.df['species'] == i]
            plt.scatter(subset['petal length (cm)'],
                        subset['petal width (cm)'],
                        label=self.species_names[i],
                        color=colors[i])
        plt.title('Scatter Plot: Petal Length vs Petal Width')
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Scatter Plot displayed successfully.")

    def histogram(self):
        colors = ['#2980B9', '#1ABC9C', '#8E44AD']
        for i in range(3):
            subset = self.df[self.df['species'] == i]
            plt.hist(subset['sepal length (cm)'],
                     label=self.species_names[i],
                     color=colors[i],
                     alpha=0.6,
                     bins=15)
        plt.title('Histogram: Sepal Length Distribution')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Histogram displayed successfully.")

    def boxplot(self):
        data_to_plot = [self.df[self.df['species'] == i]['petal length (cm)'] for i in range(3)]
        plt.boxplot(data_to_plot, labels=self.species_names)
        plt.title('Boxplot: Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        plt.tight_layout()
        plt.show()
        print("Boxplot displayed successfully.")

    def analyze_petal_vs_species(self):
        print("\n--- Petal Length Analysis per Species ---")
        for i in range(3):
            subset = self.df[self.df['species'] == i]['petal length (cm)']
            print(f"{self.species_names[i]}:")
            print(f"   Mean  : {subset.mean():.2f} cm")
            print(f"   Min   : {subset.min():.2f} cm")
            print(f"   Max   : {subset.max():.2f} cm")
        print("\nObservations:")
        print("   - Setosa has the shortest petals and is the easiest to classify.")
        print("   - Versicolor and Virginica are similar, but Virginica is larger.")
        print("   - Petal Length is a strong feature for classification.")