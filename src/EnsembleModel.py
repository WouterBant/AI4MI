from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn


@dataclass
class ModelAccuracy:
    """
    Dataclass to store model accuracies for each class.
    """
    model: nn.Module
    class_accuracies: List[float]


class EnsembleSegmentationModel(nn.Module):
    def __init__(self, model_accuracies: List[ModelAccuracy]):
        """
        Initialize the ensemble segmentation model.
        
        :param model_accuracies: List of ModelAccuracy objects containing models and their class accuracies
        """
        super(EnsembleSegmentationModel, self).__init__()

        # Extract models and create ModuleList
        self.models = nn.ModuleList([ma.model for ma in model_accuracies])

        # Extract the class accuracies for each model
        self.class_accuracies = [ma.class_accuracies for ma in model_accuracies]
        
        # Ensure all models have the same number of classes
        num_classes = len(self.class_accuracies[0])
        assert all(len(class_accuracy) == num_classes for class_accuracy in self.class_accuracies), "All models must have the same number of classes"
        
        # Initialize normalized weights
        self.normalized_weights = self._get_weights()
    
    @staticmethod
    def _get_weights(models : List[nn.Module], class_accuracies: List[List[float]], num_classes: int) -> torch.Tensor:      
        # Create a weight tensor based on class accuracies
        accuracy_weights = torch.zeros(len(models), num_classes)
        for i, accuracies in enumerate(class_accuracies):
            for class_idx, accuracy in accuracies:
                accuracy_weights[i, class_idx] = accuracy
        
        # Normalize weights
        normalized_weights = accuracy_weights / accuracy_weights.sum(dim=0, keepdim=True)
        
        return normalized_weights
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ensemble model.
        
        :param input_tensor: Input tensor
        :return: Combined prediction
        """
        # Get predictions from all models
        model_predictions = [model(input_tensor) for model in self.models]

        # Stack predictions along a new dimension
        stacked_predictions = torch.stack(model_predictions, dim=0)

        normalized_weights = self.normalized_weights.to(input_tensor.device)
        
        # Apply weights to predictions
        weighted_predictions = stacked_predictions * normalized_weights.view(len(self.models), 1, 1, self.num_classes)
        
        # Sum weighted predictions
        combined_prediction = weighted_predictions.sum(dim=0)
        
        return combined_prediction

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction and return the class with highest probability.
        
        :param input_tensor: Input tensor
        :return: Predicted class indices
        """
        with torch.no_grad():
            output = self.forward(input_tensor)
            return output.argmax(dim=1)