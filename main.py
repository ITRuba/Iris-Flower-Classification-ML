from data_handler import DataHandler
from data_visualizer import DataVisualizer
from model_trainer import ModelTrainer
from iris_predictor import IrisPredictor, IrisGUI

if __name__ == "__main__":
  handler = DataHandler()
  df_ready = handler.load_data()
  handler.show_summary()
  handler.clean_data()

  visualizer = DataVisualizer(df_ready)
  visualizer.scatter_plot()
  visualizer.histogram()
  visualizer.boxplot()
  visualizer.analyze_petal_vs_species()

  trainer = ModelTrainer(df_ready)
  trainer.split_data()
  trainer.train_model()
  trainer.evaluate_model()

  predictor = IrisPredictor(trainer.model)
  gui = IrisGUI(predictor)
  gui.run()
