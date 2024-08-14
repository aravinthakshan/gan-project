import torch
import torch.nn as nn
import torch.nn.functional as F

def gan_loss(discriminator_output, target_is_real):
    target_tensor = torch.ones_like(discriminator_output) if target_is_real else torch.zeros_like(discriminator_output)
    loss = F.binary_cross_entropy_with_logits(discriminator_output, target_tensor)
    return loss

def generator_loss(discriminator_output):
    return gan_loss(discriminator_output, target_is_real=True)

class PerceptionLoss(nn.Module):
    def __init__(self, feature_extractor, layers):
        super(PerceptionLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers

    def forward(self, generated_images, target_images):
        gen_features = self.feature_extractor(generated_images)
        target_features = self.feature_extractor(target_images)
        loss = 0
        for layer in self.layers:
            loss += F.l1_loss(gen_features[layer], target_features[layer])
        return loss

def pixel_loss(generated_images, target_images):
    return F.l1_loss(generated_images, target_images)