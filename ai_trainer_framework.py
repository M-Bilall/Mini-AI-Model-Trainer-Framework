"""Mini AI Model Trainer Framework

This module demonstrates core OOP concepts in Python:
- **Class attribute** (`BaseModel.model_count`) – tracks total models created
- **Instance attributes** (`ModelConfig.learning_rate`, `model.config`, etc.)
- **Abstraction** – `BaseModel` is an abstract base class (ABC) with abstract methods
- **Single inheritance** – `LinearRegressionModel` and `NeuralNetworkModel` inherit from `BaseModel`
- **Method overriding** – each concrete model implements its own `train`/`evaluate`
- **super()** – child ``__init__`` calls the parent ``BaseModel`` initializer
- **Polymorphism** – ``Trainer.run`` works with any ``BaseModel`` subclass
- **Composition** – a ``BaseModel`` *owns* a ``ModelConfig`` instance
- **Aggregation** – ``Trainer`` receives a ``DataLoader`` instance created elsewhere
- **Magic method ``__repr__``** – provides a readable config string
"""

from abc import ABC, abstractmethod
from typing import List

# ----------------------------
# OOP Concept: Instance Attribute
# ----------------------------
class ModelConfig:
    """Configuration object used by models (Composition).

    Attributes:
        model_name (str): Human‑readable name of the model.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
    """

    def __init__(self, model_name: str, learning_rate: float = 0.01, epochs: int = 10):
        # Store instance attributes (instance attribute concept)
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __repr__(self) -> str:
        # Magic method __repr__ – returns formatted config string
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"

# --------------------------------
# OOP Concept: Class Attribute + ABC
# --------------------------------
class BaseModel(ABC):
    """Abstract base class for all models (Abstraction)."""

    # Class attribute – shared across all subclasses
    model_count = 0

    def __init__(self, config: ModelConfig):
        # Composition: store the ModelConfig instance inside the model
        self.config = config
        # Increment the class‑level counter (demonstrates class attribute usage)
        BaseModel.model_count += 1

    @abstractmethod
    def train(self, data: List[float]):
        """Train the model on the provided data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data: List[float]):
        """Evaluate the model on the provided data."""
        raise NotImplementedError

# ------------------------------------
# OOP Concept: Single Inheritance & super()
# ------------------------------------
class LinearRegressionModel(BaseModel):
    """Concrete Linear Regression model.

    Demonstrates method overriding and use of ``super()`` to initialise the
    ``BaseModel`` part of the object.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 10):
        # Create a ModelConfig and pass it to the parent initializer (composition + super())
        config = ModelConfig(model_name="LinearRegression", learning_rate=learning_rate, epochs=epochs)
        super().__init__(config)

    # Method overriding – custom training logic
    def train(self, data: List[float]):
        print(f"LinearRegression: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
        # (Dummy training logic – real implementation omitted for brevity)

    # Method overriding – custom evaluation logic
    def evaluate(self, data: List[float]):
        # Dummy MSE calculation (placeholder value)
        mse = 0.042
        print(f"LinearRegression: Evaluation MSE = {mse}")

# -----------------------------------
# OOP Concept: Inheritance, Overriding, Composition, Additional Attribute
# -----------------------------------
class NeuralNetworkModel(BaseModel):
    """Simple feed‑forward Neural Network model.

    Adds an extra ``layers`` attribute to showcase additional state.
    """

    def __init__(self, layers: List[int] = None, learning_rate: float = 0.001, epochs: int = 20):
        if layers is None:
            layers = [64, 32, 1]
        self.layers = layers  # Instance attribute specific to NeuralNetworkModel
        config = ModelConfig(model_name="NeuralNetwork", learning_rate=learning_rate, epochs=epochs)
        super().__init__(config)

    def train(self, data: List[float]):
        layers_str = ", ".join(str(l) for l in self.layers)
        print(f"NeuralNetwork [{layers_str}]: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
        # Dummy training – nothing actually happens

    def evaluate(self, data: List[float]):
        # Dummy accuracy metric (placeholder)
        accuracy = 91.5
        print(f"NeuralNetwork: Evaluation Accuracy = {accuracy}%")

# -----------------------------------
# OOP Concept: Aggregation (external DataLoader)
# -----------------------------------
class DataLoader:
    """Simple data loader that aggregates a dataset.

    The loader is created outside of the ``Trainer`` and then passed in –
    demonstrating aggregation.
    """

    def __init__(self, dataset: List[float]):
        self.dataset = dataset

    def get_data(self) -> List[float]:
        return self.dataset

# -----------------------------------
# OOP Concept: Polymorphism (Trainer works with any BaseModel)
# -----------------------------------
class Trainer:
    """Orchestrates the training pipeline.

    Accepts any ``BaseModel`` subclass and a ``DataLoader`` instance.
    """

    def __init__(self, model: BaseModel, loader: DataLoader):
        self.model = model
        self.loader = loader

    def run(self):
        data = self.loader.get_data()
        # Polymorphic call – actual method depends on the concrete model class
        self.model.train(data)
        self.model.evaluate(data)

# -------------------------------------------------
# Demo script – produces the expected formatted output
# -------------------------------------------------
if __name__ == "__main__":
    # Create a simple dataset (aggregation example)
    loader = DataLoader(dataset=[1.0, 2.5, 3.3, 4.8, 5.2])

    # Instantiate models (composition & class attribute demonstration)
    lr_model = LinearRegressionModel(learning_rate=0.01, epochs=10)
    nn_model = NeuralNetworkModel(layers=[64, 32, 1], learning_rate=0.001, epochs=20)

    # Print configuration using the __repr__ magic method
    print(lr_model.config)
    print(nn_model.config)

    # Show total models created (class attribute)
    print(f"Models created: {BaseModel.model_count}")

    # Train & evaluate each model via the Trainer (polymorphism)
    print("--- Training LinearRegression --")
    Trainer(lr_model, loader).run()

    print("--- Training NeuralNetwork --")
    Trainer(nn_model, loader).run()
