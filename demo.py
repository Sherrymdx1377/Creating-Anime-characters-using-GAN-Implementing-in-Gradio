import requests
import pickle
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn as nn
import gradio as gr

# Set the URL of the trained model file
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0GD8EN/G_trained.pth"

# Download the trained model file
response = requests.get(url)

# Save the downloaded file locally
with open("G_trained.pth", "wb") as f:
    f.write(response.content)

latent_vector_size = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

G = Generator()

device = torch.device('cpu')
G.load_state_dict(torch.load("G_trained.pth", map_location=device))

def make_image(a, b, value):
    shape = 1
    z = a * torch.randn(1, latent_vector_size, 1, 1) + b
    Xhat = G(z)[0].detach().squeeze(0)
    Xhat = (Xhat - Xhat.min()) / (Xhat.max() - Xhat.min())
    image = to_pil_image(Xhat)
    width, height = image.size
    fixed_size = 1000  # Set your desired fixed size here
    new_width = fixed_size
    new_height = fixed_size
    resized_image = image.resize((new_width, new_height))
    resized_image.save("my_image.png")
    return resized_image

title = "Anime Creation App"
css = ".output_image {height: 60rem !important; width: 100% !important;}"

gr.Interface(
    fn= make_image,
    inputs=[
        gr.Slider(1, 10, label='Variation', value=1),
        gr.Slider(-5, 5, label='Bias', value=0),
        gr.Slider(-5, 5, label='Fine Tune: Latent Variable Value', value=0),
    ],
    title= title,
    css= css,
    outputs=gr.Image(elem_id="output_image", scale=400)
).launch()