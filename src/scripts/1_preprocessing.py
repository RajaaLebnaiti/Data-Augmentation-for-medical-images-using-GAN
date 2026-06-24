""" 
Encapsule la création des DataLoaders dans une fonction get_data_loaders(train_path, test_path, batch_size=32)

Cette fonction renvoie train_data_loader, test_data_loader 

ainsi que le dictionnaire des classes (train_dataset.class_to_idx) qui sera indispensable plus tard pour savoir quel index (0, 1, 2, 3) 
correspond à quelle tumeur 
"""

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os

def get_data_loaders(train_path, test_path, batch_size= 32):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Le dossier {train_path} n'existe pas!")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Le dossier {test_path} n'existe pas!")
    # -------------------------------------------------------------------------

    # 1. redimensionnement en 64 * 64 et normalisation entre -1 et 1
    train_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
    ])
    
    # -------------------------------------------------------------------------
    
    # 2. Regroupement des données
    train_dataset = ImageFolder(
        train_path,
        transform= train_transforms
    )
    
    test_dataset = ImageFolder(
        test_path,
        transform= test_transforms
    )
    
    # -------------------------------------------------------------------------
    
    # 3. création du dataloader
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle= True
    )
    
    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size
    )
    
    
    # -------------------------------------------------------------------------
    
    # 4. Dictionnaire des labels avec numéro
    assert train_dataset.class_to_idx == test_dataset.class_to_idx, "Erreur: les classes d'entraînement et de test ne correspondent pas !"
    dict_labels_idx = train_dataset.class_to_idx
    
    return train_dataloader, test_dataloader, dict_labels_idx