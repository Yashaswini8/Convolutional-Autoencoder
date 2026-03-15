# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
**Problem Statement:**
The goal of this experiment is to design and implement a **Convolutional Autoencoder** using **PyTorch** to perform image denoising. The model takes a noisy image as input and learns to reconstruct the clean version of the image by extracting important features through convolutional layers. The network is trained so that the reconstructed output closely matches the original image, thereby effectively removing noise from the input images.

**Dataset:**
The experiment uses the **MNIST Dataset**, which contains grayscale images of handwritten digits from 0 to 9. The dataset consists of 70,000 images in total, with 60,000 images for training and 10,000 images for testing. Each image has a resolution of 28×28 pixels. In this task, noise is artificially added to the images to create noisy inputs, and the autoencoder is trained to reconstruct the original clean images from these noisy inputs.


## DESIGN STEPS

### STEP 1:Import required libraries.

### STEP 2:Load and preprocess the image dataset.

### STEP 3:Add noise to the images.

### STEP 4:Build a convolutional autoencoder model.

### STEP 5:Define the loss function and optimizer.

### STEP 6:Train the model using noisy images as input and clean images as target.

### STEP 7:Test the model with noisy images.

### STEP 8:Reconstruct the denoised images.

## PROGRAM
### Name:YASHASWINI S
### Register Number:212224220123
```
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, loss function and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    
    for epoch in range(epochs):
        total_loss = 0
        
        for data in loader:
            img, _ = data   # images from dataset
            
            noisy_img = img + 0.2 * torch.randn_like(img)  # add noise
            
            optimizer.zero_grad()          # reset gradients
            output = model(noisy_img)      # forward pass
            loss = criterion(output, img)  # calculate loss
            
            loss.backward()                # backpropagation
            optimizer.step()               # update weights
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Yashaswini S                  ")
    print("Register Number:212224220123                  ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

```
## OUTPUT

### Model Summary
![Model Summary](https://github.com/Yashaswini8/Convolutional-Autoencoder/blob/main/model%20summary.png)

### Original vs Noisy Vs Reconstructed Image




## RESULT
